feature-extractor
=================

This is my final project for master degree program. I also put the report as pdf and odt format (in Bahasa)
If you are not a computer scientist, you might not want to see this. See my other popular repositories instead :)

The thesis program is about feature extraction to help decision tree gain better result.
The program will generate several features based on some scenario.
To execute the program, you can do:
```
   python Classify.py your_file.csv
```

The program uses Gramatical Evolution as it base method. Find more about it here http://www.grammatical-evolution.com/.
To achieve this, I make a python package to help. The package is located at gogenpy folder.
Please check `gogenpy/test/GA_Test.py`, `gogenpy/test/GP_Test.py`, and `gogenpy/test/GE_Test.py` to get a brief example about the usage.
The examples are about `Genetics Algorithm`, `Genetics Programming`, and `Grammatical Evolution` respectively