#coding:utf-8

"""
csv_data :
id      name                            format
# 0     ID row                          101 696 (UPK)
# 1     ID intervention                 99 334 different ids
# 2     alert reason category           9 different ids
# 3     alert reason                    113 different ids
# 4     on public roads                 [0, 1]
# 5     floor                           [-6:40] approxiamtely
# 6     location type                   196 different ids
# 7     longitude                       .6f 
# 8     latitude                        .6f
# 9     identifier of vehicule          612 different ids
# 10    vehicule type                   35 different names
# 11    rescue center of the vehicule   76 different ids
# 12    selection time                  YYYY-MM-DD HH:MM:SS.fff
# 13    selection time                  YYYYMMDD
# 14    selection time                  HHMMSS
# 15    status before the selection     ['Rentré', 'Disponible']
# 16    nb of sec since last status     int
# 17    vehicle departed from its RC    [0, 1]
# 18    longitude before departure      .6f
# 19    latitude before departure       .6f

brut_data :
id      name                            type    description
# 0     ID row                          int     UPK
# 1     ID intervention                 int     99 334 different ids
# 2     alert reason category           int     9 different ids
# 3     alert reason                    int     130 different ids
# 4     on public roads                 bool    [0, 1]
# 5     floor                           int     [-6:40] approxiamtely
# 6     location type                   int     214 different ids
# 7     longitude                       float   .6f 
# 8     latitude                        float   .6f
# 9     identifier of vehicule          int     785 different ids
# 10    vehicule type                   str     77 different names
# 11    rescue center of the vehicule   int     94 different ids
# 12    selection time                  datetime    YYYY-MM-DD HH:MM:SS.fff
# 13    status before the selection     str     ['Rentré', 'Disponible']
# 14    nb of sec since last status     int     int
# 15    vehicle departed from its RC    bool    [0, 1]
# 16    longitude before departure      float   .6f
# 17    latitude before departure       float   .6f

dataset :
id from :to     lenght  meaning
# 0     :9      9       reason category
# 9     :122    130     reason
# 122   :123    1       on public road
# 123   :124    1       floor
# 124   :320    214     location type
# 320   :322    2       lat, lon intervention
# 322   :934    785     vehicule id
# 934   :969    77      vehicule name
# 969   :1045   94      rescue center
# 1045  :1052   7       year, month, day, hour, minute, second, weekday
# 1052  :1053   1       if "Rentré"
# 1053  :1054   1       nb sec since last status
# 1054  :1055   1       if departed from its RC
# 1055  :1057   2       latitude, longitude departure

"""

import csv
import datetime
import numpy as np
from dateutil import parser
import pickle
import itertools


def id_to_one(id, lenght):
    """(3, 7) gives [0, 0, 0, 1, 0, 0, 0]"""
    x = np.zeros(lenght)
    x[id] = 1
    return x


def gps_coord(csv_row):
    r = np.zeros((2*127))
    try:
        gps_track = ','.join(csv_row[21:-2])
        str_coord = [doublet.split(',') for doublet in gps_track.split('"')[1].split(';')]
        coord = np.array([[float(l) for l in latlon] for latlon in str_coord])
        flattened_coord = coord.flatten()
        for i, coord in enumerate(flattened_coord):
            r[i]=coord
    except ValueError:
        pass
    return r


def gps_time(csv_row):
    r = np.zeros((3*127))
    try:
        gps_track = ','.join(csv_row[21:-2])
        a=gps_track.split('"')[2].split(';')
        a = [parser.parse(t.replace(',','')) for t in a]

        for i, value in enumerate(a):
            # r[7*i+0]=value.year/1000
            # r[7*i+1]=value.month/12
            # r[7*i+2]=value.day/60
            r[3*i+0]=value.hour/24
            r[3*i+1]=value.minute/60
            r[3*i+2]=value.second/60
            # r[7*i+6]=value.weekday()/7
    except ValueError:
        pass
    return r

# ___________________Convert csv into brut_data_______________________________
def csv_x_data_to_dict_x_data(csv_reader):
    x_brut_data = []
    for i, csv_row in enumerate(csv_reader):
        # Convert csv_data to brut_data
        if csv_row[6] == '': # Pas de location type renseigné
            csv_row[6] = 999
        if csv_row[5] =='100': # Pas de 100e étage à Paris
            csv_row[5] =='0'

        
        list_coord = gps_coord(csv_row)
        list_times = gps_time(csv_row)

        brut_row = {
            # useless
            "emergency vehicle selection":                      int(csv_row[0]),            # 0 : ID row (101696)
            "intervention":                                     int(csv_row[1]),            # 1 : ID intervention (99334)
            # boolean
            "intervention on public roads":                     bool(int(csv_row[4])),      # 4 : on public roads (2)
            "departed from its rescue center":                  bool(int(csv_row[17])),     # 15: vehicle departed from its rescue center (2)
            #Continue
            "longitude intervention":                           float(csv_row[7]),          # 7 : longitude (55579)
            "latitude intervention":                            float(csv_row[8]),          # 8 : latitude (51461)
            "longitude before departure":                       float(csv_row[18]),         # 16: longitude before departure (1762)
            "latitude previous departure":                      float(csv_row[19]),         # 17: latitude before departure (1652)
            "floor":                                            int(csv_row[5]),            # 5 : floor (41)
            "delta status preceding selection-selection":       int(csv_row[16]),           # 14: number of seconds before the vehicle was selected when its previous status (31769)
            "delta position gps previous departure-departure" : float(csv_row[20] if csv_row[20] != '' else 0),
            "OSRM estimated distance":                          float(csv_row[-2]),    
            "OSRM estimated duration":                          float(csv_row[-1]),
            # other
            "selection time":                                   parser.parse(csv_row[12]),  # 12: selection time (101690)
            "list coord":                                       list_coord,
            "list times":                                       list_times,
            # categories
            "alert reason category":                            int(csv_row[2]),            # 2 : alert reason category (9)
            "alert reason":                                     int(csv_row[3]),            # 3 : alert reason (113)
            "location of the event":                            int(float(csv_row[6])),     # 6 : location type (196)
            "emergency vehicle":                                int(csv_row[9]),            # 9 : identifier of vehicule (612)
            "rescue center":                                    int(csv_row[11]),           # 11: rescue center of the vehicule (76)
            "emergency vehicle type":                           csv_row[10],                # 10: vehicule type (35)
            "status preceding selection":                       csv_row[15],                # 13: status before the selection (2)
        }

        x_brut_data.append(brut_row)

        if i % 100000 == 99999:
            print(i+1, "rows converted into x_brut_data")

    return x_brut_data


def csv_y_data_to_y_brut_data(csv_reader):
    y_brut_data = []
    for i, csv_row in enumerate(csv_reader):
        brut_row = [
            #int(csv_row[0]),            # 0 : emergency vehicle selection
            int(csv_row[1]),            # 1 : delta selection-departure
            int(csv_row[2]),            # 2 : delta departure-presentation
            int(csv_row[3]),            # 3 : delta selection-presentation
            ]

        y_brut_data.append(brut_row)

        if i % 100000 == 99999:
            print(i+1, "rows converted into brut_data")
    return y_brut_data


if __name__ == '__main__':

    # ------------------Determination of categories---------------------
    categories = {
        "alert reason category":      set(),
        "alert reason":               set(),
        "location of the event":      set(),
        "emergency vehicle":          set(),
        "rescue center":              set(),
        "emergency vehicle type":     set(),
        "status preceding selection": set()
        }
    
    for segment in ['train', 'test']:
        with open('data/x_{}.csv'.format(segment), 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(csv_reader)
            x_dict_data = csv_x_data_to_dict_x_data(csv_reader)

   
        for dict_data in x_dict_data:
            for category, category_set in categories.items():
                category_set.add(dict_data[category])

    list_categories = {key: list(val) for key, val in categories.items()}

    # ----------------------------X--------------------------

    list_no_category = [
        # bollean
        "intervention on public roads",
        "departed from its rescue center",
        # continue
        "longitude intervention",
        "latitude intervention",
        "longitude before departure",
        "latitude previous departure",
        "floor",
        "delta status preceding selection-selection",
        "delta position gps previous departure-departure",
        "OSRM estimated distance",
        "OSRM estimated duration"]

    for segment in ['train', 'test']:
        with open('data/x_{}.csv'.format(segment), 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(csv_reader)
            x_dict_data = csv_x_data_to_dict_x_data(csv_reader)

        dataset = []
        for i_row, dict_row in enumerate(x_dict_data):
            dataset_row = []
            
            for key, value in dict_row.items():
                # Pour les autres
                if key in list_no_category:
                    dataset_row.append(np.array([float(value)]))
                
                elif key == 'selection time':
                    dataset_row.append(np.array([
                        value.year,
                        value.month,
                        value.day,
                        value.hour,
                        value.minute,
                        value.second,
                        value.weekday()
                        ]))

                elif key in ['list coord', 'list times']:
                    dataset_row.append(value)

                # Pour les catagories
                elif key in list_categories.keys():
                    # Si la valeur est catégorisée
                    dataset_row.append(id_to_one(
                        list_categories[key].index(value),
                        len(list_categories[key])
                    ))

            dataset.append(np.concatenate(dataset_row))
        
        print('trying to dump data/x_{}.pickle'.format(segment))
        pickle.dump(np.array(dataset).astype(np.float32), open('data/x_{}.pickle'.format(segment), 'wb'))
        print('success')


# ------------------------------------Y---------------------------------

    with open('data/y_train.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(csv_reader)
        y_brut_data = csv_y_data_to_y_brut_data(csv_reader)

    dataset = np.array(y_brut_data)

    print('trying to dump data/y_train.pickle')
    pickle.dump(dataset,open('data/y_train.pickle', 'wb'))
    print('success')