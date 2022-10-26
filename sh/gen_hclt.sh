juliacmd=julia
cmd="echo \${juliacmd} --project learn_hclt.jl --datasetname \${datasetname} --is_earlystop \
--latents \${latents} --pseudocount \${pseudocount} --batch_size \${batch_size} --save_circuit \
--p1 \${p1} --p2 \${p2} --p3 \${p3} --p4 \${p4} --n1 \${n1} --n2 \${n2} --n3 \${n3} --n4 \${n4} --dir \${dir} 2\>\&1 \
\| tee \${dir}/\${datasetname}_\${latents}_pse=\${pseudocount}_soft=\${softness}_bz=\${batch_size}_p=\${p1}_\${p2}_\${p3}_\${p4}_n=\${n1}_\${n2}_\${n3}_\${n4}.log"


for latents in 8 16 32 48 64 96 128 256; do
    for datasetname in mnist emnist_mnist emnist_letters emnist_balanced ptbchar_288; do
        softness=0.0
        dir=exp/hclt/$datasetname/$latents
        mkdir -p $dir
        for batch_size in 512; do
            n1=100; n2=100; n3=100; n4=100;
            for p1 in 0.0; do 
                for p2 in 0.9; do
                    for p3 in 0.99; do
                        for p4 in 0.999; do
                            for pseudocount in 0.01; do
                                eval $cmd
                            done
                        done 
                    done 
                done
            done
        done
    done
done


for latents in 8 16 32 48 64 96 128 256; do
    for datasetname in fashionmnist; do
        softness=0.0
        dir=exp/hclt/$datasetname/$latents
        mkdir -p $dir
        for batch_size in 256; do
            n1=100; n2=100; n3=100; n4=100;
            for p1 in 0.0; do 
                for p2 in 0.9; do
                    for p3 in 0.99; do
                        for p4 in 0.999; do
                            for pseudocount in 0.01; do
                                eval $cmd
                            done
                        done 
                    done 
                done
            done
        done
    done
done

for latents in 8 16 32 48 64 96 128 256; do
    for datasetname in ptbchar_288; do
        softness=0.0
        dir=exp/hclt/$datasetname/$latents
        mkdir -p $dir
        for batch_size in 256 512; do
            n1=100; n2=100; n3=100; n4=100;
            for p1 in 0.0; do 
                for p2 in 0.9; do
                    for p3 in 0.99; do
                        for p4 in 0.999; do
                            for pseudocount in 0.01 0.005; do
                                eval $cmd
                            done
                        done 
                    done 
                done
            done
        done
    done
done

for latents in 8 16 32 48 64 96 128; do
    for datasetname in emnist_byclass; do
        softness=0.0
        dir=exp/hclt/$datasetname/$latents
        mkdir -p $dir
        for batch_size in 512; do
            n1=100; n2=100; n3=100; n4=100;
            for p1 in 0.0; do 
                for p2 in 0.9; do
                    for p3 in 0.99; do
                        for p4 in 0.999; do
                            for pseudocount in 0.01; do
                                eval $cmd
                            done
                        done 
                    done 
                done
            done
        done
    done
done
