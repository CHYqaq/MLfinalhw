import os
import time
import gzip
import tarfile

t = time.time()
decompress_path = 'D:/routability_features_decompressed'
print('Create decompress dir')

if not os.path.exists(decompress_path):
    os.makedirs(decompress_path)
else:
    # Remove contents of decompress_path
    for root, dirs, files in os.walk(decompress_path):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))

filelist = os.walk('D:/routability_features')
for parent, dirnames, filenames in filelist:
    for filename in filenames:
        if os.path.splitext(filename)[1] == '.gz':
            filepath = os.path.join(parent, filename)
            print('Process %s' % (filename))

            # Decompress gzip file
            with gzip.open(filepath, 'rb') as f_in:
                tar_filepath = filepath.replace('.gz', '')
                with open(tar_filepath, 'wb') as f_out:
                    f_out.write(f_in.read())

            # Create corresponding decompressed directory
            decompressed_dir = parent.replace('routability_features', 'routability_features_decompressed')
            if not os.path.exists(decompressed_dir):
                os.makedirs(decompressed_dir)

            # Extract tar file
            with tarfile.open(tar_filepath, 'r') as tar:
                tar.extractall(path=decompressed_dir)

            # Remove intermediate tar file
            os.remove(tar_filepath)

print('Decompress finished in %ss' % (time.time() - t))
