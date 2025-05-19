
Researchers are using SAE latents to steer model behaviors, yet human-designed selection algorithms are unlikely to reach any sort of optimum for steering tasks such as SAE-based unlearning or helpfulness steering. Inspired by the [Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html), I have decided to research gradient-based optimization of steering vectors. It should be possible to add trained components into SAEs that act on the latents. These trained components could learn optimal values and algorithms, and if we chose their structure carefully, they can retain the interpretable properties of the SAE latent itself. I call these fine-tuning methods Interpretable Sparse Autoencoder Representation Fine Tuning or “ISaeRFT”.



See [here](https://www.lesswrong.com/posts/xZkBDAgQGqm6tvCvy/interpretable-fine-tuning-research-update-and-working) for more.