This is the repository for the final CNN project for the Ryerson course CPS8309, as taught by Dr. Neil Bruce. The files have been titled as meaningfully as possible (with cheeky names being changed to something more meaningful for the sake of the reader).  The main files, directly concerning machine learning are as follows:

- CNN_Binary.py: where the convolutional neural network for the binary classification model is created and trained.
- CNN_Classification.py: where the non-binary, 6 class model is created and trained.
- add_layers.py: contains functions that were moved out of the aforementioned files for modular programming. Admittidly, this is not as flexible as I would have liked, as I kept flip-flopping between object oriented programming, vague functional programming and in the instance of the make_dirs files, procedural.

To ensure that you meet all of the requirements when running these programs, ensure you run the command ```pip install -r requirements.txt ``` in your terminal.

The dataset is... questionable. Even the JSON file that contains the links to the images is not only extremely messy (thanks, LabelBox!) with tonnes of nested JSON lists, but it is still hard to access without proper permissions (again, thanks, LabelBox!). So included in submission is, ideally, a .zip containing all of the organized and scaled down images.

The comments in any and all code were omitted in the paper, as with lines of commented out code. I have tried to leave all of this as in tact as possible, but in doing so, it has proven rather messy. Was it worth it? Good question.
