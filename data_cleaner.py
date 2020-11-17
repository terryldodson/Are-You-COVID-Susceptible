import csv

with open('COVID-19_Case_Surveillance_Public_Use_Data.csv', 'r') as inp, open('clean_data.csv', 'w', newline='') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        write_row = True

        if (row[4] == 'Missing' or row[4] == 'Unknown'):
            write_row = False
        if (row[6] == 'Unknown'):
            write_row = False
        if (row[7] == 'Missing' or row[7] == 'Unknown'):
            write_row = False
        if (row[8] == 'Missing' or row[8] == 'Unknown'):
            write_row = False
        if (row[9] == 'Missing' or row[9] == 'Unknown'):
            write_row = False
        if (row[10] == 'Missing' or row[10] == 'Unknown'):
            write_row = False

        if write_row:
            writer.writerow(row)
