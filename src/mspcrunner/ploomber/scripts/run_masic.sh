# use upstream[task_name] (inside curly brackets) to declare task dependencies,
# when executing the script they will be replaced by the output of such task,
# if the task generates more than one output,
# use upstream[task_name][product_key]

#run_masic.sh
"{{product}}"

mono \
 '{{masic.exe}}' \
 "/P:{{masic.paramfile}}" \
 "/O:{{masic.outputdir}}" \
 "/I:{{masic.inputfile}}"

# {{product}}


#touch 'raw'/'{{product}}'
#touch "{{product}}"
# mono \
