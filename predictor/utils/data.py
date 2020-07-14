def process_data(timesteps, data_raw, target_raw):
    
    data, temp_data_row = list(), list()

    for i in range(len(data_raw) - (timesteps - 1)):
        for j in range(timesteps):
            temp_data_row.append(list(data_raw[i + j][4:]))

        data.append(temp_data_row)
        temp_data_row = list()

    target = target_raw[timesteps - 1:, 4]
    
    return data, target