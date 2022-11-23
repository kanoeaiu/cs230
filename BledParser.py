import pandas as pd

class BLEDParser:

    def __init__(self, bled_file, call_conf:dict, max_samples:int, sampling_rate:int,
                 exclude_unlabeled:bool = True):
        
        # Process BLED file into pandas dataframe
        # param bled_file path to text file with 2kHz BLED detections in the format MARS-%Y%m%dT%H%M%SZ-2khz.wav.Table01.txt
        # param call_conf configuration settings for generating the spectrogram
        # param exclude_unlabeled exclude detections missing a string in the label or classify column
        # param max_samples maximum number of samples associated with the wav for the given bled file
        # param sampling_rate sampling rate of data detections are associated with
        # raises exception if no wav file associated with the bled file or issue with parsing
        
        self._num_detections = 0
        self._bled_file = bled_file.name
        # date_start = datetime.strptime(bled_file.name.split('.')[0], 'MARS-%Y%m%dT%H%M%SZ-2khz')

        print(f'Reading {bled_file} {call_conf}')
        self._df = pd.read_csv(bled_file.as_posix(), sep='t')

        if self._df.empty:
            print(f'Warning {bled_file} has 0 detections')
        else:
            print(f'Found {len(self._df)} detections in {bled_file}')
            self._num_detections = len(self._df)

            call_width = int(call_conf['duration_secs'] * sampling_rate)

            def call_start(start_time, end_time):
                start_samples = int(start_time * sampling_rate)
                if call_conf['center']:
                    end_samples = int(end_time * sampling_rate)
                    middle = int(start_samples + ((end_samples - start_samples) / 2))
                    return max(middle - call_width / 2, 0)
                else:
                    return start_samples

            def call_end(start_time, end_time):
                start_samples = int(start_time * sampling_rate)
                if call_conf['center']:
                    end_samples = int(end_time * sampling_rate)
                    middle = int(start_samples + ((end_samples - start_samples) / 2))
                    return min(middle + call_width / 2, max_samples)
                else:
                    return min(start_samples + call_width, max_samples)

            def has_label(call_label):
                if pd.isnull(call_label):
                    if exclude_unlabeled:
                        return False
                    else:
                        return True
                else:
                    return True

            self._df['call_start'] = self._df.apply(lambda x: call_start(x['Begin Time (s)'], x['End Time (s)']),
                                                    axis=1)
            self._df['call_end'] = self._df.apply(lambda x: call_end(x['Begin Time (s)'], x['End Time (s)']), axis=1)

            possible_label_cols = ['label', 'Label', 'classification', 'classify']
            label_col = None
            for col in possible_label_cols:
                if col in self._df:
                    unique_labels = self._df[col].unique()
                    for label in unique_labels:
                        if pd.isnull(label):
                            continue

                    # assume only one labeled column in each file
                    label_col = col
                    break
            
            if label_col is not None:
                self._df['has_label'] = self._df.apply(lambda x: has_label(x[label_col]), axis=1)
            else:
                self._df['has_label'] = self._df.apply(lambda x: False, axis=1)
            columns_keep = [label_col, 'Classification', 'image_filename', 'Begin Time (s)', 'End Time (s)', 'Selection', 'call_start', 'call_end', 'has_label']
            for c in self._df.columns:
                if c not in columns_keep:
                    self._df = self._df.drop(c, axis=1)

    @property
    def num_detections(self):
        return self._num_detections

    @property
    def data(self):
        return self._df

    @property
    def filename(self):
        return self._bled_file