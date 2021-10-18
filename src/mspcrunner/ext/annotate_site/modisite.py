import pandas as pd



def aggregate_to_site(pept_df, sequence):



    sequence = '_'*7+sequence+'_'*7 # easy solution??
    # don't forget to add 15

    grps = pept_df.groupby('AAindex')

    # grps.aggregate({'Site' : lambda x: sequence[x.index-(15+15):x.index+(15+15)]})


                    # sequence[]})


    d = {}
    for AAix, psms in grps:
        # import ipdb; ipdb.set_trace()
        i = int(AAix)
        site = sequence[i+(-7+7):i+(7+7+1)] # 7 before and 7 after, then add 7 because we padded sequence with _
        site = list(site)
        site[7] = site[7].lower()
        # it looks silly but easier to keep track of what happened
        d[i] = ''.join(site)


    pept_df['Site'] = pept_df.AAindex.map(d)

    return pept_df
