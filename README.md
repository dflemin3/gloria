## A Hierarchical Bayesian model for the remainder of the suspended 2019-2020 NHL season

---

I miss hockey, so I decided to build a simple hierarchical Bayesian model for the 2019-2020 NHL season. Since this model is fully probabilistic, I will be able to assess the offensive and defensive ability of teams. Furthermore, I can draw samples from the posterior distribution to simulate entire seasons, and the post-season, to see who is the true Cup champ.

This work is based in large part on [Baio and Blangiardo](https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf), [Daniel Weitzenfeld's great write-up](https://danielweitzenfeld.github.io/passtheroc/blog/2014/10/28/bayes-premier-league/), and a [nice application of this type of modeling to Rugby from the pymc3 docs](https://docs.pymc.io/notebooks/rugby_analytics.html).
