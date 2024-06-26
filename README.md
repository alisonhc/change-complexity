# Data and models from the paper Learning to Paraphrase Sentences to Different Complexity Levels
Here is the link to the 2023 TACL paper: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00606/118113/Learning-to-Paraphrase-Sentences-to-Different

The data and models are accessible via [this Google Drive link](https://drive.google.com/drive/folders/1FktOG3VBG7vRLbUC89mezXIUJZHTkndB?usp=sharing). 

## Explanation of terms used in the *Data* folder in Google Drive

| Term  | Explanation |
| ------------- |:-------------:|
| up      | This dataset contains the task of complexification     |
| down      | This dataset contains the task of simplification   |
| same      | This dataset contains the task of same-level paraphrasing    |
| REL      |   The relative prompting prompts are built into the dataset.  |
| ABS      | The absolute prompting prompts are built into the dataset. |
| REVERSE INP-OUT      | For training, you will need to reverse the input and output (input should be para instead of ori)   |
| ADD PREFIX SEPARATELY      | For training, you must dynamically insert the prompts, because they are not built into the dataset. For example, if you want to use the data for REL prompt simplification, you can insert "level down: " before every input sentence. If you want to do ABS prompt simplification, you need to insert the "change to level X: ", where X is the level of the output sentence. In most datasets, output level is "para_level," but if the dataset says "REVERSE INP-OUT," the output level is "ori_level" because everything is reversed. |

