// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

//! Host check of the offline RWKV layer: prepares a collection the way
//! AnkiDroid does, builds deck counts, and prints each scored card.
//!
//! usage: rwkv_offline_host <collection.anki2> <model.bin> [out.csv]

use std::env;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::time::Instant;

use anki::collection::CollectionBuilder;
use anki::services::DecksService;
use anki::services::SchedulerService;
use anki_proto::decks::DeckTreeRequest;
use anki_proto::scheduler::RwkvOfflineInstantPassStepRequest;
use anki_proto::scheduler::RwkvPrepareOfflineRequest;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() < 3 {
        return Err("usage: rwkv_offline_host <collection.anki2> <model.bin> [out.csv]".into());
    }
    let mut col = CollectionBuilder::new(&args[1]).build()?;

    if args[2] == "repair" {
        let start = Instant::now();
        let repaired = col.repair_stripped_fsrs_memory_states()?;
        println!(
            "repaired {repaired} cards in {} ms; a second pass finds {}",
            start.elapsed().as_millis(),
            col.repair_stripped_fsrs_memory_states()?
        );
        col.close(None)?;
        return Ok(());
    }

    let start = Instant::now();
    let prepared = SchedulerService::rwkv_prepare_offline(
        &mut col,
        RwkvPrepareOfflineRequest {
            model_path: args[2].clone(),
            ..Default::default()
        },
    )?;
    println!(
        "prepare: {} reviews replayed in {} ms",
        prepared.reviews_replayed,
        start.elapsed().as_millis()
    );

    let start = Instant::now();
    let again = SchedulerService::rwkv_prepare_offline(
        &mut col,
        RwkvPrepareOfflineRequest {
            model_path: args[2].clone(),
            ..Default::default()
        },
    )?;
    println!(
        "prepare again: {} reviews replayed in {} ms",
        again.reviews_replayed,
        start.elapsed().as_millis()
    );

    let start = Instant::now();
    let pass = SchedulerService::rwkv_offline_instant_pass_step(
        &mut col,
        RwkvOfflineInstantPassStepRequest::default(),
    )?;
    println!(
        "instant pass: available={} scored={} in {} ms",
        pass.available,
        pass.scored,
        start.elapsed().as_millis()
    );

    // "Rebuild RWKV State": a full replay must give the same scores again
    let before_rebuild = {
        DecksService::deck_tree(&mut col, DeckTreeRequest { now: 0 }).ok();
        all_scores(&mut col)?
    };
    let rebuilt = SchedulerService::rwkv_offline_instant_pass_step(
        &mut col,
        RwkvOfflineInstantPassStepRequest {
            restart: true,
            ..Default::default()
        },
    )?;
    let after_rebuild = all_scores(&mut col)?;
    let status = SchedulerService::rwkv_offline_instant_pass_step(
        &mut col,
        RwkvOfflineInstantPassStepRequest {
            status_only: true,
            ..Default::default()
        },
    )?;
    println!(
        "rebuild: {} reviews replayed in {} ms, {} absorbed, {} scored; scores equal before/after: {}; status: {} absorbed, {} replayed",
        rebuilt.reviews_replayed,
        rebuilt.step_micros / 1000,
        rebuilt.reviews_absorbed,
        rebuilt.scored,
        before_rebuild.len() == after_rebuild.len()
            && before_rebuild
                .iter()
                .zip(&after_rebuild)
                .all(|(a, b)| a.0 == b.0 && (a.1 - b.1).abs() < 1e-5),
        status.reviews_absorbed,
        status.reviews_replayed,
    );

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs() as i64;
    let start = Instant::now();
    let tree = DecksService::deck_tree(&mut col, DeckTreeRequest { now })?;
    println!("deck tree in {} ms", start.elapsed().as_millis());
    for deck in &tree.children {
        println!(
            "  {}: new={} learn={} review={}",
            deck.name, deck.new_count, deck.learn_count, deck.review_count
        );
    }

    if let Some(path) = args.get(3) {
        let mut out = BufWriter::new(File::create(path)?);
        writeln!(out, "card_id,retrievability")?;
        let scores = all_scores(&mut col)?;
        for (card_id, retrievability) in &scores {
            writeln!(out, "{card_id},{retrievability:.6}")?;
        }
        println!("wrote {} scores to {path}", scores.len());
    }

    if args.get(4).map(String::as_str) == Some("answer") {
        answer_check(col, &args[1], &args[2], now)?;
    }
    Ok(())
}

fn all_scores(
    col: &mut anki::collection::Collection,
) -> Result<Vec<(i64, f32)>, Box<dyn std::error::Error>> {
    let mut scores = vec![];
    for card_id in col.search_cards("-is:new", anki::search::SortMode::NoOrder)? {
        let score = SchedulerService::get_rwkv_retrievability_score(
            col,
            anki_proto::cards::CardId { cid: card_id.0 },
        )?;
        if let Some(retrievability) = score.retrievability {
            scores.push((card_id.0, retrievability));
        }
    }
    scores.sort_by_key(|(card_id, _)| *card_id);
    Ok(scores)
}

/// Answer three scored cards through the service (as AnkiDroid does), then
/// check that the incrementally updated model state gives the same scores as
/// a fresh replay of the whole history.
fn answer_check(
    mut col: anki::collection::Collection,
    col_path: &str,
    model_path: &str,
    now: i64,
) -> Result<(), Box<dyn std::error::Error>> {
    let before = all_scores(&mut col)?;
    let mut answered: Vec<i64> = before.iter().take(3).map(|(card_id, _)| *card_id).collect();
    // The card answered Good is answered once more: a same-day repeat.
    answered.push(answered[2]);
    for (index, cid) in answered.iter().enumerate() {
        let states = SchedulerService::get_scheduling_states(
            &mut col,
            anki_proto::cards::CardId { cid: *cid },
        )?;
        // Again, Hard, Good
        let (new_state, rating) = match index {
            0 => (states.again.clone(), 0),
            1 => (states.hard.clone(), 1),
            _ => (states.good.clone(), 2),
        };
        SchedulerService::answer_card(
            &mut col,
            anki_proto::scheduler::CardAnswer {
                card_id: *cid,
                current_state: states.current.clone(),
                new_state,
                rating,
                answered_at_millis: now * 1000 + index as i64,
                milliseconds_taken: 5_000,
                ..Default::default()
            },
        )?;
    }
    let pass = SchedulerService::rwkv_offline_instant_pass_step(
        &mut col,
        RwkvOfflineInstantPassStepRequest::default(),
    )?;
    DecksService::deck_tree(&mut col, DeckTreeRequest { now })?;
    let incremental = all_scores(&mut col)?;
    println!(
        "answered {answered:?}; rescored {} cards in {} ms",
        pass.scored,
        pass.step_micros / 1000
    );
    for cid in &answered {
        let find = |scores: &[(i64, f32)]| scores.iter().find(|(id, _)| id == cid).map(|(_, r)| *r);
        println!("  {cid}: {:?} -> {:?}", find(&before), find(&incremental));
    }
    col.close(None)?;

    let mut col = CollectionBuilder::new(col_path).build()?;
    let prepared = SchedulerService::rwkv_prepare_offline(
        &mut col,
        RwkvPrepareOfflineRequest {
            model_path: model_path.into(),
            ..Default::default()
        },
    )?;
    DecksService::deck_tree(&mut col, DeckTreeRequest { now })?;
    let fresh = all_scores(&mut col)?;
    let max_diff = incremental
        .iter()
        .zip(&fresh)
        .map(|((a_id, a), (b_id, b))| {
            assert_eq!(a_id, b_id);
            (a - b).abs()
        })
        .fold(0.0f32, f32::max);
    println!(
        "fresh replay of {} reviews: {} scores, incremental {} scores, max difference {max_diff:.6}",
        prepared.reviews_replayed,
        fresh.len(),
        incremental.len()
    );
    Ok(())
}
