// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

//! RWKV-Instant without the desktop's Python.
//!
//! The desktop drives `crate::rwkv::RwkvInference` from
//! `qt/aqt/rwkv_scheduler.py`. Clients that have no Python (AnkiDroid) call
//! `RwkvPrepareOffline` once the collection is open; from then on the
//! collection keeps the model state up to date with the revlog and installs the
//! same transient score maps the desktop installs, at the two places that
//! consume them: before a review queue is built, and before deck counts are
//! computed.
//!
//! Nothing here writes to cards or the revlog. If anything fails, the scores
//! are simply not installed and the standard scheduler order applies.

use std::fmt;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use anki_proto::deck_config::deck_config::config::NewCardGatherPriority;
use anki_proto::scheduler::rwkv_historical_review_inputs_response::Review;
use anki_proto::scheduler::rwkv_review_input_rows_for_cards_response::Row;
use anki_proto::scheduler::RwkvHistoricalReviewIdentity;
use anki_proto::scheduler::RwkvHistoricalReviewInputsRequest;
use anki_proto::scheduler::RwkvOfflineInstantPassProgress;
use anki_proto::scheduler::RwkvOfflineInstantPassStepRequest;
use anki_proto::scheduler::RwkvPrepareOfflineRequest;
use anki_proto::scheduler::RwkvPrepareOfflineResponse;
use rusqlite::OptionalExtension;

use super::*;
use crate::collection::RwkvReviewQueueScoreEntry;
use crate::rwkv::ReviewInput;
use crate::rwkv::RwkvInference;
use crate::scheduler::states::CardState;
use crate::scheduler::states::NormalState;

/// The desktop always loads the model with these (they only affect interval
/// outputs, which Instant does not use).
const MODEL_TARGET_RETENTION: f32 = 0.9;
const MODEL_MAX_INTERVAL_DAYS: u32 = 36_500;
const DEFAULT_TARGET_RETENTION: f32 = 0.9;
/// Scores depend on the seconds since each card's last review, so they are
/// recomputed once they are this old, even when no review happened.
const QUEUE_SCORE_MAX_AGE_SECS: i64 = 30;
const DECK_COUNT_SCORE_MAX_AGE_SECS: i64 = 120;

/// The model path of the last successful `RwkvPrepareOffline`. The runtime
/// lives in the collection state, which is lost when the collection is
/// reopened; this lets the next queue build or deck count restore it.
static REGISTERED_MODEL: Mutex<Option<PathBuf>> = Mutex::new(None);

fn registered_model() -> Option<PathBuf> {
    REGISTERED_MODEL.lock().unwrap().clone()
}

pub(crate) struct RwkvOfflineRuntime {
    model_path: PathBuf,
    inference: RwkvInference,
    /// Feature/curve state of the freshly loaded model, for full rebuilds.
    initial_cache_state: Vec<u8>,
    /// Identity of the review history the model state has absorbed.
    identity: Option<RwkvHistoricalReviewIdentity>,
    /// Collection modification time when the history was last checked.
    checked_at_mod: Option<TimestampMillis>,
    /// Bumped whenever the model state changes; invalidates cached scores.
    generation: u64,
    scopes: HashMap<DeckId, ScopeScores>,
}

impl fmt::Debug for RwkvOfflineRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RwkvOfflineRuntime")
            .field("model_path", &self.model_path)
            .field("identity", &self.identity)
            .field("generation", &self.generation)
            .finish_non_exhaustive()
    }
}

struct ScopeScores {
    generation: u64,
    days_elapsed: u32,
    computed_at: TimestampSecs,
    cards: Vec<ScoredCard>,
}

#[derive(Clone, Copy)]
struct ScoredCard {
    card_id: CardId,
    /// The deck the card is in (filtered-deck residents are never scored).
    deck_id: DeckId,
    retrievability: f32,
    target_retention: Option<f32>,
}

impl RwkvOfflineRuntime {
    fn load(model_path: PathBuf) -> Result<Self> {
        let inference = RwkvInference::load(
            model_path.clone(),
            MODEL_TARGET_RETENTION,
            MODEL_MAX_INTERVAL_DAYS,
        )?;
        let initial_cache_state = inference.cache_state();
        Ok(Self {
            model_path,
            inference,
            initial_cache_state,
            identity: None,
            checked_at_mod: None,
            generation: 0,
            scopes: HashMap::new(),
        })
    }

    fn reset(&mut self) -> Result<()> {
        self.inference.reset_warm_up_state();
        self.inference
            .restore_cache_state(&self.initial_cache_state)?;
        self.identity = None;
        self.state_changed();
        Ok(())
    }

    fn state_changed(&mut self) {
        self.generation += 1;
        self.scopes.clear();
    }
}

fn answered_input(review: &Review) -> ReviewInput {
    ReviewInput {
        card_id: review.card_id,
        note_id: Some(review.note_id),
        deck_id: Some(review.deck_id),
        preset_id: Some(review.preset_id),
        is_query: false,
        ease: Some(review.ease as u8),
        duration_millis: Some(review.duration_millis),
        card_type: Some(review.card_type),
        day_offset: Some(review.day_offset),
        current_elapsed_days: Some(review.elapsed_days),
        current_elapsed_seconds: Some(review.elapsed_seconds),
        target_retentions: [None; 4],
        enforce_grade_order: true,
    }
}

fn valid_probability(value: f32) -> Option<f32> {
    (value.is_finite() && (0.0..=1.0).contains(&value)).then_some(value)
}

fn query_input(row: &Row) -> ReviewInput {
    let target = valid_probability(row.target_retention).unwrap_or(DEFAULT_TARGET_RETENTION);
    let card_type = if row.current_state_kind == "filtered" {
        4
    } else {
        match row.current_normal_state_kind.as_str() {
            "new" => 0,
            "learning" => 1,
            "review" => 2,
            "relearning" => 3,
            _ => row.card_type as i64,
        }
    };
    ReviewInput {
        card_id: row.card_id,
        note_id: Some(row.note_id),
        deck_id: Some(row.deck_id),
        // Add-on preset ids that are not numbers have no stable id here; the
        // history replay refuses those collections, so this is never reached
        // with one.
        preset_id: row.preset_id.parse().ok(),
        is_query: true,
        ease: None,
        duration_millis: None,
        card_type: Some(card_type),
        day_offset: Some(row.day_offset as i64),
        current_elapsed_days: row.current_elapsed_days.map(i64::from),
        current_elapsed_seconds: row.current_elapsed_seconds.map(i64::from),
        target_retentions: [Some(target); 4],
        enforce_grade_order: row.enforce_grade_order.unwrap_or(true),
    }
}

impl Collection {
    pub(crate) fn rwkv_prepare_offline(
        &mut self,
        input: RwkvPrepareOfflineRequest,
    ) -> Result<RwkvPrepareOfflineResponse> {
        require!(!input.model_path.is_empty(), "missing RWKV model path");
        let model_path = PathBuf::from(input.model_path);
        let mut runtime = match self.state.rwkv_offline.take() {
            Some(runtime) if runtime.model_path == model_path => runtime,
            _ => Box::new(RwkvOfflineRuntime::load(model_path)?),
        };
        // On error the runtime is dropped, and the standard scheduler applies
        // until the next successful prepare.
        let reviews_replayed = self.rwkv_offline_sync_history(&mut runtime)?;
        if reviews_replayed > 0 {
            self.state.card_queues = None;
        }
        *REGISTERED_MODEL.lock().unwrap() = Some(runtime.model_path.clone());
        self.state.rwkv_offline = Some(runtime);
        Ok(RwkvPrepareOfflineResponse { reviews_replayed })
    }

    /// Scores every Instant-enabled deck now and installs the deck count
    /// scores. The work is not split into steps: `complete` is always true.
    ///
    /// `restart` is the desktop's "Rebuild RWKV State": the model state is
    /// thrown away and the whole review history is replayed. `status_only`
    /// never rebuilds; it reports the state as it is (after absorbing any new
    /// reviews, which is cheap).
    pub(crate) fn rwkv_offline_instant_pass_step(
        &mut self,
        input: RwkvOfflineInstantPassStepRequest,
    ) -> Result<RwkvOfflineInstantPassProgress> {
        let Some(mut runtime) = self.state.rwkv_offline.take() else {
            return Ok(RwkvOfflineInstantPassProgress::default());
        };
        let started = Instant::now();
        let rebuild = input.restart && !input.status_only;
        let mut reviews_replayed = 0;
        let result = (|| {
            if rebuild {
                runtime.reset()?;
                runtime.checked_at_mod = None;
            }
            reviews_replayed = self.rwkv_offline_sync_history(&mut runtime)?;
            self.rwkv_offline_install_deck_count_scores_inner(&mut runtime, 0)
        })();
        let reviews_absorbed = runtime
            .identity
            .as_ref()
            .map_or(0, |identity| identity.review_count);
        self.state.rwkv_offline = Some(runtime);
        if rebuild {
            // the retained queue was ordered with the old state's scores
            self.state.card_queues = None;
        }
        let scored = result? as u32;
        Ok(RwkvOfflineInstantPassProgress {
            available: true,
            complete: true,
            scored,
            total: scored,
            step_cards: scored,
            step_micros: started.elapsed().as_micros() as u64,
            reviews_absorbed,
            reviews_replayed,
            ..Default::default()
        })
    }

    /// Called before a review queue is built.
    pub(crate) fn rwkv_offline_before_queue_build(&mut self, deck_id: DeckId) {
        self.with_rwkv_offline_runtime("queue build", |col, runtime| {
            if !col.rwkv_offline_deck_uses_instant(deck_id)? {
                return Ok(());
            }
            col.rwkv_offline_sync_history(runtime)?;
            let Some(deck) = col.storage.get_deck(deck_id)? else {
                return Ok(());
            };
            let cards = col.rwkv_offline_scope_scores(runtime, &deck, QUEUE_SCORE_MAX_AGE_SECS)?;
            let entries = col.rwkv_offline_score_entries(&deck, &cards)?;
            col.set_rwkv_review_queue_score_entries(deck_id, entries)
        });
    }

    /// Called after a card was answered: the retained queue was ordered with
    /// scores that predate the answer, so the next fetch rebuilds it.
    pub(crate) fn rwkv_offline_after_answer(&mut self) {
        if !self.rwkv_offline_enabled() {
            return;
        }
        let deck_id = self.get_current_deck_id();
        if matches!(self.rwkv_offline_deck_uses_instant(deck_id), Ok(true)) {
            self.state.card_queues = None;
        }
    }

    /// The revlog kind the desktop records when a review card is answered
    /// again on the day of its previous answer (`CardAnswer.rwkv_review_kind`,
    /// see `_rwkv_review_state_for_live_context` in rwkv_scheduler.py), so
    /// that reviews made here replay like reviews made on the desktop.
    pub(crate) fn rwkv_offline_same_day_review_kind(
        &mut self,
        card_id: CardId,
        current_state: &CardState,
        config_active: bool,
        answered_at: TimestampMillis,
        timing: &SchedTimingToday,
    ) -> Result<Option<u32>> {
        if !self.rwkv_offline_enabled()
            || !config_active
            || !matches!(current_state, CardState::Normal(NormalState::Review(_)))
        {
            return Ok(None);
        }
        let previous = self
            .storage
            .db
            .query_row(
                "select id, ease, type from revlog where cid = ? \
                 and ease between 1 and 4 and type in (0, 1, 2, 3, 4, 5) \
                 and not (type = 3 and factor = 0) order by id desc limit 1",
                [card_id.0],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                },
            )
            .optional()?;
        let Some((previous_id, previous_ease, previous_kind)) = previous else {
            return Ok(None);
        };
        if rwkv_historical_day_offset(previous_id, timing)
            != rwkv_historical_day_offset(answered_at.0, timing)
        {
            return Ok(None);
        }
        const RELEARNING: u32 = 2;
        const FILTERED: u32 = 3;
        Ok(match (previous_kind, previous_ease) {
            (1, 1) | (2, 1 | 2) => Some(RELEARNING),
            (0 | 1 | 2, _) => Some(FILTERED),
            _ => None,
        })
    }

    /// Called before deck counts are computed.
    pub(crate) fn rwkv_offline_before_deck_counts(&mut self) {
        self.with_rwkv_offline_runtime("deck counts", |col, runtime| {
            col.rwkv_offline_sync_history(runtime)?;
            col.rwkv_offline_install_deck_count_scores_inner(
                runtime,
                DECK_COUNT_SCORE_MAX_AGE_SECS,
            )?;
            Ok(())
        });
    }

    fn rwkv_offline_enabled(&self) -> bool {
        self.state.rwkv_offline.is_some() || registered_model().is_some()
    }

    fn with_rwkv_offline_runtime(
        &mut self,
        stage: &str,
        func: impl FnOnce(&mut Collection, &mut RwkvOfflineRuntime) -> Result<()>,
    ) {
        let mut runtime = match self.state.rwkv_offline.take() {
            Some(runtime) => runtime,
            None => {
                // The collection was reopened (e.g. after a full sync) since
                // the client's prepare call.
                let Some(model_path) = registered_model() else {
                    return;
                };
                match RwkvOfflineRuntime::load(model_path) {
                    Ok(runtime) => Box::new(runtime),
                    Err(err) => {
                        tracing::warn!(?err, "RWKV offline model failed to load; disabled");
                        *REGISTERED_MODEL.lock().unwrap() = None;
                        return;
                    }
                }
            }
        };
        if let Err(err) = func(self, &mut runtime) {
            tracing::warn!(
                ?err,
                stage,
                "RWKV offline scoring failed; using stored order"
            );
        }
        self.state.rwkv_offline = Some(runtime);
    }

    fn rwkv_offline_deck_uses_instant(&mut self, deck_id: DeckId) -> Result<bool> {
        Ok(self
            .rwkv_offline_deck_config(deck_id)?
            .is_some_and(|config| config.inner.rwkv_review_instant_order_enabled))
    }

    fn rwkv_offline_deck_config(&mut self, deck_id: DeckId) -> Result<Option<DeckConfig>> {
        let Some(config_id) = self
            .storage
            .get_deck(deck_id)?
            .and_then(|deck| deck.config_id())
        else {
            return Ok(None);
        };
        self.storage.get_deck_config(config_id)
    }

    /// Bring the model state in line with the revlog. Returns the number of
    /// reviews replayed.
    fn rwkv_offline_sync_history(&mut self, runtime: &mut RwkvOfflineRuntime) -> Result<u64> {
        let collection_mod = self.storage.get_collection_timestamps()?.collection_change;
        if runtime.identity.is_some() && runtime.checked_at_mod == Some(collection_mod) {
            return Ok(0);
        }

        let dynamic_preset_replay = self.storage.all_deck_config()?.iter().any(|config| {
            (config.inner.rwkv_review_enabled || config.inner.rwkv_review_instant_order_enabled)
                && config.inner.rwkv_review_dynamic_preset_replay
        });
        let mut history = RwkvHistoricalReviewFingerprintRequest {
            dynamic_preset_replay,
            ..Default::default()
        };

        let mut absorbed = 0;
        if let Some(identity) = runtime.identity.clone().filter(|id| id.review_count > 0) {
            history.expected_identity = Some(identity.clone());
            let fingerprint = self.rwkv_historical_review_fingerprint(history.clone())?;
            history.expected_identity = None;
            if fingerprint.history_is_valid {
                runtime.checked_at_mod = Some(collection_mod);
                return Ok(0);
            }
            if fingerprint.history_prefix_is_valid {
                absorbed = identity.review_count as usize;
            }
        }

        let response = self.rwkv_historical_review_inputs(RwkvHistoricalReviewInputsRequest {
            history: Some(history),
            ..Default::default()
        })?;
        if absorbed == 0 || absorbed > response.reviews.len() {
            absorbed = 0;
            runtime.reset()?;
        }
        let reviews: Vec<_> = response.reviews[absorbed..]
            .iter()
            .map(answered_input)
            .collect();
        let replayed = reviews.len() as u64;
        // The identity is cleared first so a failed replay forces a rebuild.
        runtime.identity = None;
        runtime.inference.warm_up_reviews(reviews, false)?;
        runtime.identity = Some(
            response
                .metadata
                .and_then(|metadata| metadata.identity)
                .unwrap_or_default(),
        );
        runtime.checked_at_mod = Some(collection_mod);
        runtime.state_changed();
        Ok(replayed)
    }

    /// Current retrievability of the scoreable cards in `deck` and its
    /// children.
    fn rwkv_offline_scope_scores(
        &mut self,
        runtime: &mut RwkvOfflineRuntime,
        deck: &Deck,
        max_age_secs: i64,
    ) -> Result<Vec<ScoredCard>> {
        let days_elapsed = self.timing_today()?.days_elapsed;
        let now = TimestampSecs::now();
        if let Some(scope) = runtime.scopes.get(&deck.id) {
            if scope.generation == runtime.generation
                && scope.days_elapsed == days_elapsed
                && now.0 - scope.computed_at.0 < max_age_secs
            {
                return Ok(scope.cards.clone());
            }
        }

        let include_new_cards = self
            .rwkv_offline_deck_config(deck.id)?
            .is_some_and(|config| {
                matches!(
                    config.inner.new_card_gather_priority(),
                    NewCardGatherPriority::DescendingRetrievability
                        | NewCardGatherPriority::AscendingRetrievability
                )
            });
        let rows = self
            .rwkv_review_input_rows_for_deck_review_queue(
                RwkvReviewInputRowsForDeckReviewQueueRequest {
                    deck_id: deck.id.0,
                    include_disabled_decks: false,
                    include_new_cards,
                },
            )?
            .rows;
        let inputs: Vec<_> = rows.iter().map(query_input).collect();
        let retrievabilities = runtime
            .inference
            .predict_retrievability_many_from_warm_up(inputs)?;
        let cards: Vec<_> = rows
            .iter()
            .zip(retrievabilities)
            .filter_map(|(row, retrievability)| {
                Some(ScoredCard {
                    card_id: CardId(row.card_id),
                    deck_id: DeckId(row.deck_id),
                    retrievability: valid_probability(retrievability)?,
                    target_retention: valid_probability(row.target_retention),
                })
            })
            .collect();
        runtime.scopes.insert(
            deck.id,
            ScopeScores {
                generation: runtime.generation,
                days_elapsed,
                computed_at: now,
                cards: cards.clone(),
            },
        );
        Ok(cards)
    }

    /// Score entries for `cards`, with the repeat spacing the desktop derives
    /// from the most recent reviews in `deck`'s tree.
    fn rwkv_offline_score_entries(
        &mut self,
        deck: &Deck,
        cards: &[ScoredCard],
    ) -> Result<HashMap<CardId, RwkvReviewQueueScoreEntry>> {
        let intervening = self.rwkv_offline_intervening_reviews(deck)?;
        Ok(cards
            .iter()
            .map(|card| {
                (
                    card.card_id,
                    RwkvReviewQueueScoreEntry {
                        retrievability: card.retrievability,
                        intervening_reviews: intervening.get(&card.card_id).copied(),
                        target_retention: card.target_retention,
                    },
                )
            })
            .collect())
    }

    /// For cards among the last `min_intervening_reviews` answers in the
    /// deck's tree: how many answers came after the card's latest one.
    fn rwkv_offline_intervening_reviews(&mut self, deck: &Deck) -> Result<HashMap<CardId, u32>> {
        let limit = self
            .rwkv_offline_deck_config(deck.id)?
            .map(|config| config.inner.rwkv_review_min_intervening_reviews)
            .unwrap_or_default();
        let mut out = HashMap::new();
        if limit == 0 {
            return Ok(out);
        }
        let deck_ids = self.rwkv_offline_deck_tree_ids(deck)?;
        let deck_ids: Vec<_> = deck_ids.iter().map(|id| id.0.to_string()).collect();
        let sql = format!(
            "select r.cid from revlog r join cards c on c.id = r.cid \
             where r.ease between 1 and 4 and r.type in (0, 1, 2, 3, 4, 5) \
             and not (r.type = 3 and r.factor = 0) \
             and (case when c.odid != 0 then c.odid else c.did end) in ({}) \
             order by r.id desc, r.cid desc limit ?",
            deck_ids.join(",")
        );
        let mut stmt = self.storage.db.prepare(&sql)?;
        let rows = stmt.query_map([limit], |row| row.get::<_, i64>(0))?;
        for (index, card_id) in rows.enumerate() {
            out.entry(CardId(card_id?)).or_insert(index as u32);
        }
        Ok(out)
    }

    fn rwkv_offline_deck_tree_ids(&mut self, deck: &Deck) -> Result<HashSet<DeckId>> {
        let mut ids: HashSet<_> = self
            .storage
            .child_decks(deck)?
            .into_iter()
            .map(|child| child.id)
            .collect();
        ids.insert(deck.id);
        Ok(ids)
    }

    /// Install deck count scores for every top-most Instant-enabled deck and
    /// its children, like the desktop's deck browser. Returns the number of
    /// cards scored.
    fn rwkv_offline_install_deck_count_scores_inner(
        &mut self,
        runtime: &mut RwkvOfflineRuntime,
        max_age_secs: i64,
    ) -> Result<usize> {
        let configs = self.storage.get_deck_config_map()?;
        let mut decks = self.storage.get_all_decks()?;
        decks.sort_by(|a, b| a.name.as_native_str().cmp(b.name.as_native_str()));
        let mut scopes: Vec<Deck> = Vec::new();
        for deck in decks {
            let enabled = deck
                .config_id()
                .and_then(|config_id| configs.get(&config_id))
                .is_some_and(|config| config.inner.rwkv_review_instant_order_enabled);
            let covered = scopes.iter().any(|scope| {
                deck.name
                    .as_native_str()
                    .strip_prefix(scope.name.as_native_str())
                    .is_some_and(|rest| rest.starts_with('\x1f'))
            });
            if enabled && !covered {
                scopes.push(deck);
            }
        }

        self.clear_rwkv_deck_count_scores();
        let mut scored = 0;
        for scope in scopes {
            let cards = self.rwkv_offline_scope_scores(runtime, &scope, max_age_secs)?;
            scored += cards.len();
            let entries = self.rwkv_offline_score_entries(&scope, &cards)?;
            self.set_rwkv_deck_count_score_entries(scope.id, entries)?;
            // Predictions are shared with the children, but repeat spacing
            // must use each child's own recent answers.
            for child in self.storage.child_decks(&scope)? {
                let tree = self.rwkv_offline_deck_tree_ids(&child)?;
                let child_cards: Vec<_> = cards
                    .iter()
                    .filter(|card| tree.contains(&card.deck_id))
                    .copied()
                    .collect();
                let entries = self.rwkv_offline_score_entries(&child, &child_cards)?;
                self.set_rwkv_deck_count_score_entries(child.id, entries)?;
            }
        }
        Ok(scored)
    }
}
