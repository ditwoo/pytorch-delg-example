
## Implementation of [Unifying Deep Local and Global Features for Image Search](https://arxiv.org/abs/2001.05027)


From paper above it's important to train model in two steps - training global features & training local features.

Global features produce embedding like this:

<p align="center">
    <img src="misc/global_features_embedding.png">
</p>

(Big black points - class centroids)

For training local features used attention, here are few examples of attention maps:

<p align="center" float="left">
    <img src="misc/attention_heatmap_0.png" width="250">
    <img src="misc/attention_heatmap_1.png" width="250">
    <img src="misc/attention_heatmap_2.png" width="250">
</p>
