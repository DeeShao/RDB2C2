# RDB2C2
stand-alone software package for RDB2C2

There are two steps to use our software RDB2C2.
Let's use 1a3a protein as an example and put the sequence file under the file folder 'sequence': ./rdb2c2/sequence/1a3a
STEP 1:
prepare all the feature we are going to use.
(1) ccmpred supply two files and put them in the file folder 'ccmpred': ./rdb2c2/ccmpred/1a3a.aln ./rdb2c2/ccmpred/1a3a.ccmpred
(2) Deepconpred2 supply one contact map file and put it in the file folder 'deepconpred2': ./rdb2c2/deepconpred2/1a3a_contactmap.txt
(3) Spider3 supply the secondary structures file and put it in the file folder 'Spider3': ./rdb2c2/Spider3/1a3a.spd33
put all the files in the right file folder and all the file format are shown with the 1a3a protein as an example.
STEP 2:
run the commandline: bash pipeline.sh 1a3a
Results:
The result files are in the file folder '.rdb2c2/TF-feature/result/: .rdb2c2/TF-feature/result/1a3a_map.png .rdb2c2/TF-feature/result/1a3a_result.txt.
