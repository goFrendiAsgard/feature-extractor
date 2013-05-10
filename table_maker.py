dir_name = 'RESULT/ecoli.edit (5 fold)'
label_list = ['GA Select Feature', 'GE Global', 'GE Multi', 'GE Tatami Multi']
fold_count = 5

import os

# make table head
thead = '<td colspan="2">Percobaan</td>'
for label in label_list:
    thead += '<td>'+label+'</td>'
thead = '<thead><tr>'+thead+'</tr></thead>'

# make table body
tbody = ''
for fold_index in xrange(1,fold_count+1):
    training_row, testing_row, total_row = '', '', ''
    for label in label_list:
        file_name = os.path.join(dir_name, label+' Fold '+str(fold_index)+'.txt')
        fh = open(file_name)
        accuracy = []
        for lino, line in enumerate(fh):
            if lino>4:
                break
            elif lino >= 2:
                value = float(line.split(':')[1].lstrip()) * 100.0
                value = round(value, 2)
                accuracy.append(str(value)+'%')
        training_row += '<td>'+accuracy[0]+'</td>'
        testing_row += '<td>'+accuracy[1]+'</td>'
        total_row += '<td>'+accuracy[2]+'</td>'
        fh.close()
    # tbody    
    training_row = '<tr><td rowspan="3">Fold ' + str(fold_index) + '</td><td>Training</td>' + training_row + '</tr>'    
    testing_row = '<tr><td>Testing</td>' + testing_row + '</tr>'    
    total_row = '<tr><td>Total</td>' + total_row + '</tr>'
    tbody += training_row + testing_row + total_row

html = '<table>'+thead+tbody+'<table>'
# write the file
html_file_name = os.path.join(dir_name,'resume.html')
fh = open(html_file_name, 'w')
fh.write(html)
fh.close()