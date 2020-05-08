# Applications in Earth Science

Personally, I haven't seen too many applications in Earth sciences. I've seen it show up explicitly in two scenarios: inverse problems and density estimation of the Earth.


---

## Literature


* **Normalizing Flows on Tori and Spheres** - Rezende et. al. (2020) - [arxiv](https://arxiv.org/abs/2002.02428) 
    > They use normalizing flows for density estimation on different complex geometrices (e.g. Toris or Spheres). They have an example where they do it on the Earth. Not much explanation but it looked really cool.
* **Analyzing Inverse Problems with Invertible Neural Networks** - Ardizzone et. al. (2019) - [arxiv](https://arxiv.org/abs/1808.04730)
    > They use Normalizing flows (what they call invertible neural networks) to solve inverse problems. They also have a nice [toolbox](https://github.com/VLL-HD/FrEIA) and [reproducible code](https://github.com/VLL-HD/analyzing_inverse_problems) for this paper.
* **Hybrid Models with Deep and Invertible Features** - Nalisnick et. al. (2019) - [plmr](http://proceedings.mlr.press/v97/nalisnick19b.html)
    > While this is not directly applied to Earth sciences, I think this is a really cool idea. To get a probability distribution, train a normalizing flow on data. And then couple that to a machine learning model. Super neat! No code though. Couldn't find anything...