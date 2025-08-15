# SD.Next Remote-TE Experiment

- use `serve.py` to run the server
- use `test.py` to test the server
- set `SD_REMOTE_T5="http://127.0.0.1:7850/te"` to use inside SD.Next  

considerations:

- use Nunchaku `SVDQuant` instead of `BitsAndBytes`, its about 2-3x faster  
- host using dedicated engine, not pure `transformers`  
- add other common text-encoders: umt5, llama, qwen-2.5, etc.
