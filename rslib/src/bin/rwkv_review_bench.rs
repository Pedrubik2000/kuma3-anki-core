// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

//! Times the reviewer's answer -> next card loop with offline RWKV-Instant,
//! as AnkiDroid runs it. Use a copy of the collection: it answers cards Good.
//!
//! usage: rwkv_review_bench <collection.anki2> <model.bin> <deck name>
//! [answers] (RUST_LOG=info shows the offline layer's own timings)

use std::env;
use std::time::Instant;

use anki::collection::Collection;
use anki::collection::CollectionBuilder;
use anki::services::ConfigService;
use anki::services::SchedulerService;
use anki_proto::scheduler::CardAnswer;
use anki_proto::scheduler::GetQueuedCardsRequest;
use anki_proto::scheduler::QueuedCards;
use anki_proto::scheduler::RwkvOfflineInstantPassStepRequest;
use anki_proto::scheduler::RwkvPrepareOfflineRequest;

fn fetch(col: &mut Collection) -> anki::error::Result<QueuedCards> {
    SchedulerService::get_queued_cards(
        col,
        GetQueuedCardsRequest {
            fetch_limit: 1,
            ..Default::default()
        },
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    anki::log::set_global_logger(None)?;
    let args: Vec<_> = env::args().collect();
    if args.len() < 4 {
        return Err(
            "usage: rwkv_review_bench <collection.anki2> <model.bin> <deck> [answers]".into(),
        );
    }
    let answers: usize = args.get(4).map_or(Ok(10), |n| n.parse())?;
    let mut col = CollectionBuilder::new(&args[1]).build()?;
    let _ = SchedulerService::rwkv_prepare_offline(
        &mut col,
        RwkvPrepareOfflineRequest {
            model_path: args[2].clone(),
        },
    )?;
    while !SchedulerService::rwkv_offline_instant_pass_step(
        &mut col,
        RwkvOfflineInstantPassStepRequest {
            status_only: true,
            ..Default::default()
        },
    )?
    .available
    {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let deck = col.get_deck_id(&args[3])?.ok_or("no such deck")?;
    col.set_current_deck(deck)?;

    let started = Instant::now();
    let mut next = fetch(&mut col)?;
    println!("first fetch: {} ms", started.elapsed().as_millis());
    let mut times = vec![];
    for _ in 0..answers {
        let Some(queued) = next.cards.first() else {
            break;
        };
        let card_id = queued.card.as_ref().unwrap().id;
        let states = queued.states.clone().unwrap();
        let started = Instant::now();
        let _ = SchedulerService::answer_card(
            &mut col,
            CardAnswer {
                card_id,
                current_state: states.current,
                new_state: states.good,
                rating: 2,
                answered_at_millis: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_millis() as i64,
                milliseconds_taken: 5_000,
                ..Default::default()
            },
        )?;
        let answered = started.elapsed().as_millis();
        next = fetch(&mut col)?;
        let total = started.elapsed().as_millis();
        println!(
            "answer {card_id}: answer {answered} ms, next card {} ms",
            total - answered
        );
        times.push(total);
    }
    times.sort();
    if !times.is_empty() {
        println!(
            "answer -> next card: median {} ms, max {} ms",
            times[times.len() / 2],
            times[times.len() - 1]
        );
    }
    // Answers were appended to the model state one by one; a full check of
    // the history must find nothing to replay.
    let _ = ConfigService::set_config_bool(
        &mut col,
        anki_proto::config::SetConfigBoolRequest {
            key: anki_proto::config::config_key::Bool::BrowserTableShowNotesMode as i32,
            value: true,
            undoable: false,
        },
    )?;
    let check = SchedulerService::rwkv_offline_instant_pass_step(
        &mut col,
        RwkvOfflineInstantPassStepRequest {
            status_only: true,
            ..Default::default()
        },
    )?;
    println!(
        "full history check after the answers: {} reviews replayed (0 = appended state matches), {} absorbed",
        check.reviews_replayed, check.reviews_absorbed
    );
    Ok(())
}
