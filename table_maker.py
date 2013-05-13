import os

def make_resume(data_name, fold_count):
    if fold_count == 1:
        dir_name = 'RESULT/'+data_name+' (whole)'
    else:
        dir_name = 'RESULT/'+data_name+' ('+str(fold_count)+' fold)'
    label_list = ['GA Select Feature', 'GE Global', 'GE Multi', 'GE Tatami Multi', 'GE Gravalis']
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
    
    # make table body
    tbody = ''
    for fold_index in xrange(1,fold_count+1):
        training_row, testing_row, total_row = '', '', ''
        for label in label_list:
            file_name = os.path.join(dir_name, label+' Fold '+str(fold_index)+'.txt')
            fh = open(file_name)
            accuracy = []
            feature_used = 0
            for lino, line in enumerate(fh):
                if lino>7:
                    break
                elif lino >= 2 and lino<=4:
                    value = float(line.split(':')[1].lstrip()) * 100.0
                    value = round(value, 2)
                    accuracy.append(str(value)+'%')
                elif lino == 7:
                    feature_used = str(int(line.split(' ')[0]))
            training_row += '<td>'+accuracy[0]+'</td><td rowspan="3">'+feature_used+'</td>'
            testing_row += '<td>'+accuracy[1]+'</td>'
            total_row += '<td>'+accuracy[2]+'</td>'
            fh.close()
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

data_name_list = ['iris.data','balance-scale.data','ecoli.edit','synthesis_01','synthesis_02','synthesis_03']
for data_name in data_name_list:
    make_resume(data_name,1)
    make_resume(data_name,5)