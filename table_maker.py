import os
import numpy as np
label_list = ['GA Select Feature', 'GE Global', 'GE Multi', 'GE Tatami Multi', 'GE Gravalis']

def make_resume(data_name, fold_count):
    if fold_count == 1:
        dir_name = 'RESULT/'+data_name+' (whole)'
    else:
        dir_name = 'RESULT/'+data_name+' ('+str(fold_count)+' fold)'
    
    # make table head
    thead = '<tr>'
    thead += '<th>Experiment</th><th>&nbsp;</th>'
    for label in label_list:
        thead += '<th>'+label+'</th><th>&nbsp;</th>'
    thead += '</tr>'
    thead += '<tr>'
    thead += '<th>&nbsp;</th><th>&nbsp;</th>'
    for label in label_list:
        thead += '<th>Acc</th>'
        thead += '<th>F. Count</th>'
    thead += '</tr>'
    thead = '<thead>'+thead+'</thead>'
    
    result_list = []
    # make table body
    tbody = ''
    for fold_index in xrange(1,fold_count+1):
        result = {}
        training_row, testing_row, total_row = '', '', ''
        for label in label_list:
            file_name = os.path.join(dir_name, label+' Fold '+str(fold_index)+'.txt')
            fh = open(file_name)
            accuracy = []
            accuracy_value = []
            feature_used = 0
            for lino, line in enumerate(fh):
                if lino>7:
                    break
                elif lino >= 2 and lino<=4:
                    value = float(line.split(':')[1].lstrip()) * 100.0
                    value = round(value, 2)
                    accuracy_value.append(value)
                    accuracy.append(str(value)+'%')
                elif lino == 7:
                    feature_used = str(int(line.split(' ')[0]))
            training_row += '<td>'+accuracy[0]+'</td><td rowspan="3">'+feature_used+'</td>'
            testing_row += '<td>'+accuracy[1]+'</td>'
            total_row += '<td>'+accuracy[2]+'</td>'
            result[label] = [accuracy_value, int(feature_used)]
            fh.close()
        result_list.append(result)
        # tbody
        training_row = '<tr><td rowspan="3">Fold ' + str(fold_index) + '</td><td>Training</td>' + training_row + '</tr>'
        testing_row = '<tr><td>Testing</td>' + testing_row + '</tr>'
        total_row = '<tr><td>Total</td>' + total_row + '</tr>'
        tbody += training_row + testing_row + total_row
    last_row = '<tr><td colspan="'+str(2+2*len(label_list))+'">end of table</td></tr>'
    tbody = '<tbody>'+tbody+last_row+'</tbody>'
    html = '<table border="1">'+thead+tbody+'<table>'
    # write the file
    html_file_name = os.path.join(dir_name,'resume.html')
    fh = open(html_file_name, 'w')
    fh.write(html)
    fh.close()
    return result_list

data_name_list = ['iris.data','balance-scale.data','ecoli.edit','synthesis_01','synthesis_02','synthesis_03']
resume = {}
for data_name in data_name_list:    
    # get whole accuracy & perfold accuracy
    perfold_result = make_resume(data_name,5)
    whole_result = list(make_resume(data_name, 1))[0]
    
    summary = {}
    for key, value in whole_result.iteritems():
        summary[key] = [ [[],[],[]], [] ]
        summary[key][0][0].append(whole_result[key][0][0])
        summary[key][0][1].append(whole_result[key][0][1])
        summary[key][0][2].append(whole_result[key][0][2])
        summary[key][1].append(whole_result[key][1])
        for result in perfold_result:
            summary[key][0][0].append(result[key][0][0])
            summary[key][0][1].append(result[key][0][1])
            summary[key][0][2].append(result[key][0][2])
            summary[key][1].append(result[key][1])
    
    resume[data_name] = summary

# make html
tr = ''
all_resume = {} 
for data_name in data_name_list:
    first_row  = '<td>'+data_name+'</td><td>Training</td>'
    second_row = '<td>&nbsp;</td><td>Test</td>'
    third_row  = '<td>&nbsp;</td><td>Global</td>'
    for key in label_list:
        value = resume[data_name][key]
        # add to all_resume
        if key not in all_resume:
            all_resume[key] = [ [[],[],[]], [] ]
        all_resume[key][0][0].append(value[0][0])
        all_resume[key][0][1].append(value[0][1])
        all_resume[key][0][2].append(value[0][2])
        all_resume[key][1].append(value[1])
        # make html
        accuracy = value[0]
        feature_used = value[1]
        first_row  += '<td>'+str(round(np.mean(accuracy[0]),2))+'%</td><td>'+str(round(np.std(accuracy[0]),2))+'</td><td>'+ \
            str(round(np.mean(feature_used)))+'</td>'
        second_row += '<td>'+str(round(np.mean(accuracy[1]),2))+'%</td><td>'+str(round(np.std(accuracy[1]),2))+'</td><td>&nbsp;</td>'
        third_row  += '<td>'+str(round(np.mean(accuracy[2]),2))+'%</td><td>'+str(round(np.std(accuracy[2]),2))+'</td><td>&nbsp;</td>'
    tr += '<tr>'+first_row+'<tr>'+second_row+'</tr><tr>'+third_row+'</tr>'



# all_resume
first_row  = '<td>All</td><td>Training</td>'
second_row = '<td>&nbsp;</td><td>Test</td>'
third_row  = '<td>&nbsp;</td><td>Global</td>'
for key in label_list:
    value = all_resume[key]
    accuracy = value[0]
    feature_used = value[1]
    first_row  += '<td>'+str(round(np.mean(accuracy[0]),2))+'%</td><td>'+str(round(np.std(accuracy[0]),2))+'</td><td>'+ \
        str(round(np.mean(feature_used)))+'</td>'
    second_row += '<td>'+str(round(np.mean(accuracy[1]),2))+'%</td><td>'+str(round(np.std(accuracy[1]),2))+'</td><td>&nbsp;</td>'
    third_row  += '<td>'+str(round(np.mean(accuracy[2]),2))+'%</td><td>'+str(round(np.std(accuracy[2]),2))+'</td><td>&nbsp;</td>'
tr += '<tr>'+first_row+'<tr>'+second_row+'</tr><tr>'+third_row+'</tr>'
    
html_header = ''
for item in label_list:
    html_header += '<td>'+item+'</td><td>&nbsp;</td><td>&nbsp;</td>'
html = '<table border="1"><tr><td>1</td><td>2</td>'+html_header+'</tr>'+tr+'</table>'
print html

file_name = 'RESULT/resume.html'
fh = open(file_name, 'w')
fh.write(html)
fh.close()