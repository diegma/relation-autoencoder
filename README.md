# relation-autoencoder
This is the code used in the paper [Discrete-State Variational Autoencoders for Joint Discovery and Factorization of Relations](https://transacl.org/ojs/index.php/tacl/article/viewFile/761/190) by Diego Marcheggiani and Ivan Titov.

If you use this code, please cite us.

Dependencies
-----------
- [theano](http://deeplearning.net/software/theano/)
- [numpy](http://http://www.numpy.org/)
- [scipy](http://https://www.scipy.org/)
- [nltk](http://http://www.nltk.org/)


Data Processing
--------------
To run the model the first thing to do is create a dataset.
You need a file like data-sample.txt.
The file must be tab-separated an with the following fields:

lexicalized dependency path between arguments (entities) of the relation,
first entity
second entity
entity types of the first and second entity
trigger word
id of the sentence
raw sentence
pos tags of the entire sentence
relation between the two entities if any (used only for evaluation)


In order to create the dataset you need the OiePreprocessor.py script once for each dataset partition: train, dev, and test.
<pre><code>
python processing/OiePreprocessor.py --batch-name train data-sample.txt sample.pk 
python processing/OiePreprocessor.py --batch-name dev data-sample.txt sample.pk
python processing/OiePreprocessor.py --batch-name test data-sample.txt sample.pk
</code></pre>

Now, your dataset with all the indexed features is in sample.pk

Training Models
------------
To train the model run the OieInduction.py file with all the required arguments:
<pre><code>
python learning/OieInduction.py --pickled_dataset sample.pk --model_name discrete-autoencoder --model AC --optimization 1 --epochs 10 --batch_size 100 --relations_number 10 --negative_samples_number 5 --l2_regularization 0.1 --alpha 0.1 --seed 2 --embed_size 10 --learning_rate 0.1
</code></pre>


For any questions, please drop me a mail at marcheggiani [at] uva [dot] nl. 
