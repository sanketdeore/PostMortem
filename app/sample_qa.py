"""Ten labeled answers used by `python -m app.compare_models`."""

SAMPLE_QA = [
    (
        "Tell me about a time you sped up a service.",
        "Um, it was slow, you know, and I kind of looked around and basically made it better I think.",
    ),
    (
        "How would you design a rate limiter?",
        "I would design the limiter as a token bucket in Redis: 100 requests per minute per API key, and return 429 when the bucket is empty.",
    ),
    (
        "Describe a production incident you owned.",
        "The checkout error rate hit 8 percent. I rolled back the last deploy in 12 minutes and error rate went back to 0.4 percent.",
    ),
    (
        "Why this company?",
        "I like the office and my friend works nearby so the commute would be easy.",
    ),
    (
        "Tell me about a disagreement with a teammate.",
        "We disagreed about the schema. I wrote a one-page comparison, we picked the smaller table, and the migration finished in two days instead of the two weeks the wider design needed.",
    ),
    (
        "What is your greatest weakness?",
        "I work too hard.",
    ),
    (
        "Walk me through a project you are proud of.",
        "So like the project was, um, a dashboard and I did the backend and also some of the frontend and there were meetings and then we shipped it and people said it was nice and I learned a lot which was good.",
    ),
    (
        "How do you test a payments change?",
        "First I write a failing test for the rounding bug. Then I fix the integer-cents path. The result was zero mismatched settlements across 10,000 replayed orders.",
    ),
    (
        "Tell me about a time you mentored someone.",
        "I helped them.",
    ),
    (
        "Estimate capacity for this API.",
        "It depends on a lot of factors and there are many ways to think about capacity and one must consider the future and the past and the team and the cloud and the users and the budget and the roadmap and the stakeholders and the latency and the errors and the regions and the cache and the database and the queue and the clients and the mobile apps and the web and the partners and the SLAs and the on-call rotation and the documentation and the hiring plan.",
    ),
]
