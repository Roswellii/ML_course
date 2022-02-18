import xlsxwriter as xw
with open("train_acc") as file:
    lines= file.readlines()
    file.close()
workbook= xw.Workbook('.train.xlsx')
sheet= workbook.add_worksheet('sheet1')
sheet.write(0,0, 'train loss')
sheet.write(1,0, 'train acc')
sheet.write(2,0, 'validation loss')
sheet.write(3,0, 'validation acc')
i= 1
for line in lines:
    if "train loss" in line:
        line= line.replace('\n', '')
        data= line.split(':')[1]
        sheet.write(0, i, data)
    if "train acc" in line:
        line= line.replace('\n', '')
        data= line.split(':')[1]
        sheet.write(1, i, data)
    if "validation loss" in line:
        line= line.replace('\n', '')
        data= line.split(':')[1]
        sheet.write(2, i, data)
    if "validation acc" in line:
        line= line.replace('\n', '')
        data= line.split(':')[1]
        sheet.write(3, i, data)
        i+=1
workbook.close()