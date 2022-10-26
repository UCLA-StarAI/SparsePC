# SparsePC
Code of the NeurIPS 2022 paper [Sparse Probabilistic Circuits via Pruning and Growing](http://starai.cs.ucla.edu/papers/DangNeurIPS22.pdf).


## Environment
1. Install Juilia version 1.8.
2. Run the following command to install required packages.
   ```
   julia env.jl
   ```

## Datasets
Run the following command to download all datasets. And enter `y` each time when hinted to.
   ```
   julia --project download_data.jl
   ```

## Experiments

Run the following commands to generate runnable scripts.
  1. Learning hidden Chow-Liu trees (HCLT).
     ```
     bash sh/gen_hclt.sh
     ```
      After learning, postprocess checkpoints to pick the best HCLTs as initial structures and save them in `circuits\`.
     ```
     julia --project post_process.jl
     ```

  2. Learning sparse PC structure via pruning and growing (Section 6.1).
     ```
     bash sh/gen_loop.sh
     ````

  3. Distilling large PC by iteratively pruning (Section 6.2).
     ```
     bash sh/gen_compress.sh
     ```


## Citation
To cite this paper, please use 
```
@inproceedings{DangNeurIPS22,
  author    = {Dang, Meihua and Liu, Anji and Van den Broeck, Guy},
  title     = {Sparse Probabilistic Circuits via Pruning and Growing},
  booktitle = {Advances in Neural Information Processing Systems 35 (NeurIPS)},
  month     = {dec},
  year      = {2022},
}
```