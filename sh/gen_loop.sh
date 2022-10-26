juliacmd=julia

cmd="echo \$juliacmd --project loop.jl --datasetname \$datasetname --latents \$latents --by topk \
--dir \$dir --std \$std --threshold \$threshold --pseudocount \$pseudocount \
--n0 \$n0 --n \$maxiter --n1 \$n1 --n2 \$n2 --n3 \$n3 --n4 \$n4 \
--p1 \$p1 --p2 \$p2 --p3 \$p3 --p4 \$p4 --pm \$pm --warmup \$n \
--isfinetune \$isfinetune \
2\>\&1 \| tee \
\$dir/\${datasetname}_z=\${latents}_pse=\${pseudocount}_std=\${std}_thre=\${threshold}_isfinetune=\${isfinetune}_p=\${pm}_\${p1}_\${p2}_\${p3}_\${p4}_n=\${n0}_\${n1}_\${n2}_\${n3}_\${n4}.log"


for datasetname in mnist emnist_mnist emnist_letters emnist_balanced emnist_byclass fashionmnist ptbchar_288 ; do
    for latents in 8 16 32 48 64 96 128; do
        dir="exp/loop/$datasetname" 
        mkdir -p $dir
        maxiter=20
        for threshold in 0.75; do 
            for pseudocount in 0.01; do 
                for std in 0.3; do
                    for n in 50; do
                        for pm in 0.7; do
                            p1=0.0; p2=0.9; p3=0.99; p4=0.999
                            n1=$n; n2=$n; n3=$n; n4=0
                            eval $cmd
                        done
                    done
                done
            done
        done
    done
done