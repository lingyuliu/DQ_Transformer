# Look, Compare and Draw: Differential Query Transformer for Automatic Oil Painting

> [[Project Page](https://differential-query-painter.github.io/DQ-painter/)]

This work has been accepted by *IEEE Transactions on Visualization and Computer Graphics*, 2026. 

### Abstract

> This work introduces a new approach to automatic
> oil painting that emphasizes the creation of dynamic and
> expressive brushstrokes. A pivotal challenge lies in mitigating
> the duplicate and common-place strokes, which often lead to less
> aesthetic outcomes. Inspired by the human painting process, i.e.,
> observing, comparing, and drawing, we incorporate differential
> image analysis into a neural oil painting model, allowing the
> model to effectively concentrate on the incremental impact
> of successive brushstrokes. To operationalize this concept, we
> propose the Differential Query Transformer (DQ-Transformer),
> a new architecture that leverages differentially derived image
> representations enriched with positional encoding to guide the
> stroke prediction process. This integration enables the model
> to maintain heightened sensitivity to local details, resulting in
> more refined and nuanced stroke generation. Furthermore, we
> incorporate adversarial training into our framework, enhancing
> the accuracy of stroke prediction and thereby improving the
> overall realism and fidelity of the synthesized paintings. Extensive
> qualitative evaluations, complemented by a controlled user study,
> validate that our DQ-Transformer surpasses existing methods in
> both visual realism and artistic authenticity, typically achieving
> these results with fewer strokes. The stroke-by-stroke painting
> animations are available on our project website.



## Prerequisites

* Linux or macOS
* Python 3.9
* PyTorch 1.7+ and other dependencies (torchvision, visdom, dominate, and other common python libs)

## Training

```shell
  cd train
  bash my_train.sh
```

* models would be saved at checkpoints/painter folder.

## Inference

```shell
  cd inference
  python inference.py
```
## Citation

If you find this code helpful for your research, please cite:

```
@article{liu2026look,
author = "Liu, Lingyu and Wang, Yaxiong and Zhu, Li and Liao, Lizi and Zheng, Zhedong",
title = "Look, Compare and Draw: Differential Query Transformer for Automatic Oil Painting",
journal = "TVCG",
code = "https://differential-query-painter.github.io/DQ-painter/",
year = "2026" }
```

## Acknowledgments

This repository is benefit from [Paint Transformer](https://github.com/Huage001/PaintTransformer). Thanks for the
open-sourcing work! We would also like to thank to the great projects
in [Compositional Neural Painter](https://github.com/sjtuplayer/Compositional_Neural_Painter).