# Example images

| File | Content | Source |
|---|---|---|
| `images/ccmt_cassava_blight_on_paper.png` | cassava bacterial blight, leaf photographed on paper | CCMT dataset, Ghana |
| `images/ccmt_cassava_blight_in_field.png` | cassava bacterial blight, leaf photographed in the field | CCMT dataset, Ghana |

Both are resized crops of images from the CCMT dataset: Mensah, P. K., et al. (2023). CCMT:
Dataset for crop pest and disease detection. *Data in Brief*, 49, 109306.
https://doi.org/10.17632/bwh3zbpkpv.1 — licensed CC BY 4.0.

They come from the external Ghanaian source, where the models are weakest, so a low confidence or
an abstention on these images is consistent with the published results rather than a fault.

```bash
python scripts/predict.py --crop cassava examples/images/*.png
```
