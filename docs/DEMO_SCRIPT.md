# Demo video script (~3 minutes)

**0:00 Problem (20 s)** "When one microservice breaks, dependent services fail one after another. Teams see the crash, not the cause. CascadeGuard predicts the cascade, finds the root cause and picks the best fix before everything goes down."

**0:20 Setup (15 s)** Show sidebar: "We secretly break `orders-db` with a connection-pool fault. The AI isn't told. We look 6 minutes later."

**0:35 Overview tab (25 s)** Graph: only `orders-db` is red. Latency chart + error logs show the first symptoms.

**1:00 Root cause tab (30 s)** "It ranks `orders-db` first: earliest, explains the other sick services via the graph, healthy dependencies, its own errors. Logs show 'connection pool exhausted' so it diagnoses connection-pool exhaustion."

**1:30 Prediction tab (30 s)** "Monte-Carlo cascade: `orders`, `payments`, `gateway` will fail within minutes - before it has happened."

**2:00 Recovery tab (25 s)** "We simulate each fix. Failover cuts damage ~50 % vs doing nothing." Change fault to *bad deploy* → recommendation switches to *rollback*.

**2:25 Explanation tab (15 s)** Ollama ON: generate the briefing. "Works without Ollama too - automatic template fallback."

**2:40 Validation + honesty (20 s)** Reveal truth: root and fault found, cascade predicted. "Data is simulated; next step is validating on real traces."
