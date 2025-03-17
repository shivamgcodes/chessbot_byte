class data_config:

    batch_size = 1 # guess
    shuffle = False
    number_of_files = 3 #max is 2148
    filename = f'/home/shivam/Desktop/chessbot_byte/data/train/action_value-@{number_of_files}_data.bag'
    num_return_buckets = 128 # in the repo
    miniDataSet = False

    if(miniDataSet):
        count = 100
    
