for x in $@; do
    echo "Working on $x"
    jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.kernel_name=cmomy-dev --execute $x
    echo "done"
done
