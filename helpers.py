def download_wav(utcstring):
  """
  Grab a wav file for a given utc string, e.g. 20200711T000000Z
  """
  date = parser.parse(utcstring)
  wav_filename = f'MARS-{date:%Y%m%dT%H%M%SZ}-16kHz.wav'
  bucket = 'pacific-sound-16khz'
  key = f'{date.year:04d}/{date.month:02d}/{wav_filename}'

  s3 = boto3.resource('s3',
      aws_access_key_id='',
      aws_secret_access_key='',
      config=Config(signature_version=UNSIGNED))
  
  # only download if needed
  if not Path(wav_filename).exists():

      print(f'Downloading {key} from s3://{bucket}') 
      s3.Bucket(bucket).download_file(key, wav_filename)
      print('Done')