"""rlcombine

read multiple summary loss streams from multiple rl inuring priorities and combine

the input is the summary loss stream from each individual IP for a given summaryset (and process)

i.e. fifo/rl_IP1_S1_combine_P1, fifo/rl_IP2_S1_combine_P2, ... fifo/rl_IPN_S1_combine_P2

The summary streams from each individual IP are structured something like this:

header │ E1:S2 │ E1:S5 │ E1:S9 │ E7:S1 │ E7:S5 │ E3:S2 │ …
         └──── event 1 ────┘     └─ event 7 ─┘  └ event 3 …

STREAM HEADER (12 B, once): [agg type=1 (3 B) | source type=3 (1 B)] int32 len_sample=N | int32 summary_set_id

and each summary record (e.g. E1:S2) consists of:
    single header (12 bytes): int32 event_id, int32 summary_id, float32 expval
    multiple records (8 bytes each): (int32 sidx, float32 loss)
        sidx -5, -4, -1 are the max, affected risk and mean analytical loss respectively
        sidx of a positive integar is a single sample
        sidx 0 = terminator (end of summary record)

To combine summary streams from multiple IP we need to know when a particular
inuring priority has sent the final event_id record, including when an event_id
has no loss and no records. Events with no loss are never sent. Hence we need a
sentinel record to notify when an `event_id` has finished. This should only
happen if --combine-out is passed to summarypy and this type of stream will
only be passed to this module (not the output components) and will be stripped
from this stream in this module before passing on. Propose using a `summary_id`
of 0 for the sentinel record for example `E10:S0` and the sidx 0 terminator
afterwards to preserve consistency with the other parsers. Then you can know
when a record is done for a particular `event_id` without reading the next
record in the stream.

We can use the fact that that the event_ids should be aligned and the
summary_ids should be sorted as they get passed to rlcombine to do a merge with
these streams. #TODO - check this

The pseudo code for this module looks something like:

open inputs; read the 12-byte header of each
    all must have source type == SUMMARY (3), agg type == ITEM (1),
    and equal len_sample and summary_set_id               → else error
write output header (same three values); out_offset = 12

for each input k: head[k] = next_record(k)                  # non-blocking fill, see below
if every head is EOF: done                                   # empty run
if any head is EOF: error

loop:
    E = head[0].event_id
    if any head[k].event_id != E: error (event sequence mismatch)

    S = min(head[k].summary_id, with S0 = ∞)

    if S == ∞:                                               # every input has closed event E
        for each k: head[k] = next_record(k)
        if every head is EOF: done
        if any head is EOF: error
        continue

    acc = new summary record (E, S)
    for each k with head[k].summary_id == S:
        combine head[k] into acc
        prev[k] = S
        head[k] = next_record(k)
        if head[k] is EOF: error (event E not closed by S0)
        if head[k] is not S0:
            if head[k].event_id != E: error (event E not closed by S0)
            if head[k].summary_id <= prev[k]: error (summary_id not strictly increasing)

    if -z or acc.tiv != 0 or acc.mean != 0:
        if -m: write (S, out_offset) to .idx
        write acc to output buffer (flushing and splitting across 64 KB chunks)
        out_offset += record size

next_record(k):
    while stream k has no complete record (terminator not yet buffered) and not EOF:
        select on all open inputs; for each ready input j, read what is available into buf[j]
    parse one record from buf[k] (or return EOF)

Method of combining records depends on the type of the field:

    sidx 1...N         | regular sample                  | sum
    sidx -1            | mean                            | sum (no. samples are the same)
    sidx -2            | std dev                         | not in summary stream → error if seen
    header expval (-3) | TIV (passed through fm terms)   | sum
    sidx -4            | chance of loss / affected risks | not written for -t ri → error if seen
    sidx -5            | max loss                        | sum


# TODO: check that tiv and max loss are actually calculated per term in the RI case and not just passed through.
    - what to do with these fields will depend on this output.
"""
