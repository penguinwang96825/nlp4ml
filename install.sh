# Download pre-trained embedding inside embedding folder
if [ ! -d "./embedding" ]
then
	mkdir ./embedding
fi

cd ./embedding

if [ ! -e "./GoogleNews-vectors-negative300.bin.gz" ]
then
	wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
fi

if [ ! -e "./glove.840B.300d.zip" ]
then
	wget http://nlp.stanford.edu/data/glove.840B.300d.zip
fi

if [ ! -e "./cc.en.300.bin.gz" ]
then
	wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
fi

if [ -e "./GoogleNews-vectors-negative300.bin.gz" ]
then
	gzip -d GoogleNews-vectors-negative300.bin.gz
fi

if [ -e "./glove.840B.300d.zip" ]
then
	unzip glove.840B.300d.zip
fi

if [ -e "./cc.en.300.bin.gz" ]
then
	gzip -d cc.en.300.bin.gz
fi