# NirNet: End2End Sentence-level Lipreading
A novel method that apply deep learning into lip reading

## Results
|       Scenario          | Epoch |  CER  |  WER  |  BLEU |
|:-----------------------:|:-----:|:-----:|:-----:|:-----:|
|  Unseen speakers [C]    |  N/A  |  N/A  |  N/A  |  N/A  |
|    Unseen speakers      |  N/A  |  N/A  |  0.226  |  N/A  |
| Overlapped speakers [C] |  N/A  |  N/A  |  N/A  |  N/A  |
|   Overlapped speakers   |  N/A  |  N/A  |  N/A  |  N/A  |

## Dependencies
* PyTorch 0.5 (With nn.CTCLoss)

## Usage

Modify options.toml and 

```
python main.py options.toml
``` 

## Dataset

This model uses GRID corpus (http://spandh.dcs.shef.ac.uk/gridcorpus/)
Please extract mouth region using dlib or other tools.
