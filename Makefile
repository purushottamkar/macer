install:
	# Install required tools
	apt-get install clang python3-pip unzip gzip curl sqlite3
	if [ -f /usr/lib/x86_64-linux-gnu/libclang.so ] ; then echo "libclang symbolic link exists"; else ln -s /usr/lib/x86_64-linux-gnu/libclang-*.so.1 /usr/lib/x86_64-linux-gnu/libclang.so; fi
	
	# Initialize conda environment and install required libraries
	conda create --name macer36 python=3.6
	conda activate macer36
	pip3 install --version -r requirements.txt

	# Pull Tracer dataset
	git clone https://github.com/umairzahmed/tracer.git
	unzip tracer/data/dataset/singleL/singleL_Test.zip -d tracer/data/dataset/singleL/
	unzip tracer/data/dataset/singleL/singleL_Train+Valid.zip -d tracer/data/dataset/singleL/

	# Create classes from Tracer dataset
	python3 -m srcT.DataStruct.ClusterError

	# Pull Deepfix dataset
	curl -O https://www.cse.iitk.ac.in/users/karkare/prutor/prutor-deepfix-09-12-2017.zip
	unzip prutor-deepfix-09-12-2017.zip
	gzip -d prutor-deepfix-09-12-2017/prutor-deepfix-09-12-2017.db.gz

	# Extract dataset into Macer's format
	sqlite3 -header -csv prutor-deepfix-09-12-2017/prutor-deepfix-09-12-2017.db "select * from Code where error<>'';" > prutor-deepfix-09-12-2017/deepfix_test.csv
	python3 -m srcT.DataStruct.PrepDeepFix
