## The cache stays lossless

Every check ran on the L4 with Qwen3-8B and the KV tables the engine made for it; the outputs are in `checks/`. A value the cache holds is the bits that were written, and where the cache has no room for what a request needs, the
request is never answered from what it could not hold.

- **A known answer** (`checks/kvtest.out`, `checks/kvtest-first-tables.out`). bf16 values made on the host, of every kind a cache is given (inside the range its layer's windows hold and outside it, both zeros, subnormals,
  infinities, NaNs of several payloads, the largest and the smallest numbers, blocks with few, many and more exceptions than a block can keep in its own room, and the 65,536 bit patterns once each) were written into 15 blocks of 16 tokens of each of
  the 36 layers, as one run of tokens, token by token and in pieces, read back and compared bit for bit; then written again from the first token and again from the middle, and compared again. 53,084,160 values compared
  (319 zeros, 317 negative zeros, 16,928 subnormals, 631 infinities, 17,452 NaNs in each of the three passes, a fifth of the values outside their windows): **0 differ**. With every block released the cache's extra memory
  is empty (0 of 28196 and 0 of 256 in use), and 0 blocks were lost.
- **Every write compared with a bf16 copy of what was written, at the step that makes it** (a copy of every layer's cache; each step's tokens are decoded back and compared before the next step runs): requests of natural text and of
  random tokens, 4 at a time on a cache of 128 to 144 blocks, so that finished prompts are reused, kept for later prompts and found again, and requests wait and are run again (the table below; `checks/churn-*.out`,
  `checks/extra-memory-*.out`, `checks/write-check-*.out`). In the server (`checks/churn-server-natural.out`): 300 short requests of natural text from 4 clients on a cache of 128 blocks, 0 wrong, 226,579 prompt tokens,
  5,680 found in the cache, 59 sent back to wait; 32,469,073,920 values compared, **0 differ**.
- **Where the cache has no room for what a request needs** (extra memory cut to 16 and 2, and to 8 and 1 units, so that random tokens run it out; `checks/extra-memory-*.out`): the step in which a block could not be stored exactly is named
  at once, the output of that step is taken back from the requests that hold the block (no token is made from it), and each such request waits until the ones running with it have ended, then runs again from its own text; one that is
  taken back twice runs alone; one that is lost alone with nothing kept for later prompts ends with an error in words. A request that ends so is run again alone, to see that it does not fit alone: those that made no token end so again; one that had made tokens before it ended may be served, its text run again as a prompt being computed another way than the same tokens were by steps (its keys and values then differ in their last bit) and so needing other extra memory (the table says how many end so again). The values of blocks named lost are left out of
  the comparison (counted in the table), every other value is compared. The few requests that part from their run alone do so after being run again, at a place where the model's two best choices are near-equal: a prompt run again is
  computed in another way than the same tokens were by steps (`checks/run-again.out`), so its keys and values differ in their last bit from the ones the steps made. In the server (`checks/server-extra-memory.out`): 6 checks ok, 0 failed: a request the cache cannot hold is answered with the reason in words, one it can hold
  is served whole, and the one after the failures gets the text it got before them.
- **After many rounds of churn** (`checks/rowcheck-fresh.out`, `checks/rowcheck-churn-300.out`): one sequence's logits with 0 to 7 others in the step, at every number of rows and in either row, on a fresh cache and after
  300 rounds of 8 sequences of random lengths on the same memory (100,407,214,080 values compared, 0 differ): the same logits bit for bit in every one.

The runs of requests against a bf16 copy (each request's tokens against the same request run alone on a cache with room, where that was made):

| run | requests | users | cache tokens | values compared | differing | values in lost blocks (not compared) | steps taken back | sent back to wait | ended with an error | same as alone / part from it without / after being sent back |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| natural text, 300 requests, 128 blocks | 300 natural | 4 | 2048 | 24,889,761,792 | 0 | 0 | 0 | 15 | 0 | (not compared) |
| random tokens, 300 requests, 128 blocks | 300 random | 4 | 2048 | 17,595,113,472 | 0 | 0 | 0 | 0 | 0 | (not compared) |
| natural text and random tokens, 300 requests, 128 blocks, little extra memory (16 and 2 units) | 300 mixed | 4 | 2048 | 23,068,385,280 | 0 | 133,963,776 | 129 | 130 | 0 | (not compared) |
| random tokens, extra memory 16 and 2 units | 100 random | 4 | 2304 | 6,843,727,872 | 0 | 44,015,616 | 46 | 46 | 0 | 97 / 0 / 3 |
| random tokens, extra memory 8 and 1 units | 100 random | 4 | 2304 | 8,600,223,744 | 0 | 120,766,464 | 128 | 128 | 0 | 91 / 0 / 9 |
| random tokens, extra memory 3 and 0 units (requests that do not fit alone end with an error) | 100 random | 4 | 2304 | 9,862,963,200 | 0 | 323,665,920 | 249 | 215 | 34 (alone, 33 end so again) | 61 / 0 / 5 |
| natural text, 100 requests, room for all of them | 100 natural | 4 | 2304 | 8,624,111,616 | 0 | 0 | 0 | 0 | 0 | 100 / 0 / 0 |
| random tokens, 100 requests, room for all of them | 100 random | 4 | 2304 | 5,946,900,480 | 0 | 0 | 0 | 0 | 0 | 100 / 0 / 0 |
