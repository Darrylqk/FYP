import csv

def write_to_csv(filename, headers, data):
    with open('./csv_files/' + filename + '.csv','w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        csvwriter.writerow(data)        
