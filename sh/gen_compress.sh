juliacmd=julia

cmd="echo \$juliacmd --project samesize_loop.jl --datasetname \$datasetname --latents \$latents --by topk \
--dir \$dir --std \$std --threshold \$threshold --pseudocount \$pseudocount --batch_size \$batch_size \
--n \$maxiter --n1 \$n1 --n2 \$n2 --n3 \$n3 --n4 \$n4 \
--p1 \$p1 --p2 \$p2 --p3 \$p3 --p4 \$p4 --pm \$pm --warmup \$n --nogrow \
2\>\&1 \| tee \
\$dir/\${datasetname}_z=\${latents}_pse=\${pseudocount}_std=\${std}_thre=\${threshold}_bz=\${batch_size}_p=\${pm}_\${p1}_\${p2}_\${p3}_\${p4}_n=\${maxiter}_\${n1}_\${n2}_\${n3}_\${n4}.log"


for latents in 16 32 64 128; do
    for datasetname in emnist_byclass emnist_mnist; do
        std=0
        dir="exp/distill/$datasetname/$latents"
        mkdir -p $dir
        maxiter=20
        batch_size=512
        for threshold in 0.1 0.3 ; do
            for pseudocount in 0.01; do 
                for n in 30; do
                    pm=0.9; p1=0.9; p2=0.9; p3=0.99; p4=0.999
                    n1=0; n2=$n; n3=$n; n4=0
                    eval $cmd
                done
            done
        done
    done
done