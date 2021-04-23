from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import requests, json, time, os, zipfile, io
import girder_client
from datetime import datetime
from joblib import Parallel, delayed
from glob import glob
import numpy as np
import six
from six.moves import zip  # pylint: disable=redefined-builtin

class federated(object):
    """
    A class that provides functions for federated learning of semantic
    segmentaiton of WSIs via multiple distributed servers.
    This uses the Digital slide archive and HistomicsTK to perform training
    """

    def __init__(self):

        ### starting params ###
        self.training_steps_per_round = 100
        self.training_rounds = 2
        self.global_step = 0
        self.slow_start_step = 5
        self.patch_size = 400
        self.learning_rate = 0.0005
        self.learning_rate_start = 0.00001
        self.end_learning_rate = 0.0
        self.learning_power = 0.9
        self.wsi_downsample = [1,2,3,4]
        self.batch_size = 2
        self.fine_tune_batch_norm = False
        self.augment = 0
        self.init_last_layer = True
        self.gpu = '0'
        self.num_clones = 1
        self.classes = '["gloms"]'
        self.output_model = 'model.zip'
        self.starting_model = 'model-ImageNet-Xception.zip'

        ### server params ###
        self.server_params = []
        self.server_params.append({
                'token'                 : 'RccgHJJZDE0PiaU17HZaNJsQKqbjEeatzyiJQu9rmUtbUhYZhNjRJZoGA7R6V9zi',
                'url'                   : 'http://zeus.med.buffalo.edu:8080',
                'inputFolder'           : '60821209cef56adf6fc063be',
                'output_model_folder'   : '6082be9bcef56adf6fc06847',
                })

        self.server_params.append({
                'token'                 : 'yp89C63Cr7dKHJecN0NsiiTFx4f9dmwK1oQoq9RUVKRMJWypGMBE2TovOAnk1KL6',
                'url'                   : 'http://kronos.med.buffalo.edu:8080',
                'inputFolder'           : '6082174f19c88eb9f84a89a7',
                'output_model_folder'   : '6082be8419c88eb9f84a8bdb',
                })


    ############################################################################
    ### main functions #######################################################
    ############################################################################

    def federated_training(self):
        total_training_steps = self.training_steps_per_round * self.training_rounds

        def training_round(server_dict, starting_model):
            # upload starting model
            inputModelFile = self.upload_model(server_dict, starting_model)
            # train and download model
            model_folder = self.training_cycle(server_dict, inputModelFile)
            return model_folder

        # federated training step
        for round in range(self.training_rounds):
            round += 1
            print('\nstarting training round {}\n'.format(round))
            self.training_steps = self.training_steps_per_round * round

            # run training in parallel
            model_folders = Parallel(n_jobs=len(self.server_params))(delayed(training_round)(server_dict, self.starting_model) for server_dict in self.server_params)

            # average models
            saved_model_name = 'round-{}-averaged-model'.format(round)
            self.starting_model = self.average_and_zip_model(model_folders, saved_model_name)


    ############################################################################
    ### helper functions #######################################################
    ############################################################################

    def upload_model(self, args, file):
        ### upload model to server ###
        gc = girder_client.GirderClient(apiUrl='{}/api/v1'.format(args['url']))
        gc.setToken(args['token'])
        gc.upload(file, args['output_model_folder'], reuseExisting=True)
        id = list(gc.listItem(args['output_model_folder'], name=file))[0]['_id']
        inputModelFile = list(gc.listFile(id))[0]['_id']
        return inputModelFile

    def training_cycle(self, args, inputModelFile):
        ### train and download model ###
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Girder-Token': '{}'.format(args['token']),
        }

        now = datetime.now().strftime("%m-%d-%Y %H_%M_%S")
        base, ext = os.path.splitext(self.output_model)
        output_model_name = '{}-{}{}'.format(base, now, ext)

        params = (
            ('inputFolder', args['inputFolder']),
            ('output_model_folder', args['output_model_folder']),
            ('output_model', output_model_name),
            ('inputModelFile', inputModelFile),
            ('WSI_downsample', str(self.wsi_downsample)),
            ('augment', str(self.augment)),
            ('batch_norm', str(self.fine_tune_batch_norm)),
            ('batch_size', str(self.batch_size)),
            ('classes', self.classes),
            ('init_last_layer', str(self.init_last_layer)),
            ('learning_rate', str(self.learning_rate)),
            ('learning_rate_start', str(self.learning_rate_start)),
            ('num_clones', str(self.num_clones)),
            ('GPU', self.gpu),
            ('patch_size', str(self.patch_size)),
            ('slow_start_step', str(self.slow_start_step)),
            ('steps', str(self.training_steps)),
            ('global_step', str(self.global_step)),
            ('end_learning_rate', str(self.end_learning_rate)),
            ('learning_power', str(self.learning_power)),
            ('use_xml', 'false'),
            ('girderApiUrl', '{}/api/v1'.format(args['url'])),
        )

        response = requests.post('{}/api/v1/slicer_cli_web/brendonl_histomicstk-deeplab_GPU-accelerated-tf/TrainNetwork/run'.format(args['url']), headers=headers, params=params)

        server = args['url'].split('://')[-1]
        server = server.split('.')[0]

        body = json.loads(response.content)
        job_id = body['_id']
        print('\tjob submited on {} with id: {}'.format(server, job_id))

        ### check training status ###
        status = 0
        while status <= 2:
            time.sleep(10) # check job status every 10 seconds
            response = requests.get('{}/api/v1/job/{}'.format(args['url'], job_id), headers=headers)
            body = json.loads(response.content)
            status = body['status']
            print('\t{} | {} still training...'.format(datetime.now().strftime("%H:%M:%S"), server))

        print('\t{} done with status code: {}'.format(server, status))
        assert status==3, '!!! training job failed on {} !!!'.format(args['url'])


        ### download saved model ###

        # find model
        params = (
            ('folderId', args['output_model_folder']),
            ('name', output_model_name),
            ('limit', '50'),
            ('sort', 'lowerName'),
            ('sortdir', '1'),
        )

        response = requests.get('{}/api/v1/item'.format(args['url']), headers=headers, params=params)
        body = json.loads(response.content)
        model_id = body[0]['_id']

        # download model
        params = (
            ('contentDisposition', 'attachment'),
        )

        response = requests.get('{}/api/v1/item/{}/download'.format(args['url'], model_id), headers=headers, params=params)

        # extract models from zip
        model_path = os.path.splitext(output_model_name)[0]
        model_path = '{} - {}'.format(server,model_path)
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        z = zipfile.ZipFile(io.BytesIO(response.content))
        for name in z.namelist():
            z.extract(name, model_path)
        return model_path

    def avg_checkpoints(self, input_checkpoints="", num_last_checkpoints=0, prefix="", output_path="averaged.ckpt"):
        """Script to average values of variables in a list of checkpoint files."""
        import tensorflow.compat.v1 as tf
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=self.gpu

        # flags.DEFINE_string("checkpoints", "",
        #                     "Comma-separated list of checkpoints to average.")
        # flags.DEFINE_integer("num_last_checkpoints", 0,
        #                      "Averages the last N saved checkpoints."
        #                      " If the checkpoints flag is set, this is ignored.")
        # flags.DEFINE_string("prefix", "",
        #                     "Prefix (e.g., directory) to append to each checkpoint.")
        # flags.DEFINE_string("output_path", "/tmp/averaged.ckpt",
        #                     "Path to output the averaged checkpoint to.")

        def checkpoint_exists(path):
            return (tf.gfile.Exists(path) or tf.gfile.Exists(path + ".meta") or
                    tf.gfile.Exists(path + ".index"))


        if input_checkpoints:
            # Get the checkpoints list from flags and run some basic checks.
            checkpoints = [c.strip() for c in input_checkpoints]
            checkpoints = [c for c in checkpoints if c]
            if not checkpoints:
                raise ValueError("No checkpoints provided for averaging.")
            if prefix:
                checkpoints = [prefix + c for c in checkpoints]

        else:
            assert num_last_checkpoints >= 1, "Must average at least one model"
            assert prefix, ("Prefix must be provided when averaging last"
                                                        " N checkpoints")
            checkpoint_state = tf.train.get_checkpoint_state(
                    os.path.dirname(prefix))
            # Checkpoints are ordered from oldest to newest.
            checkpoints = checkpoint_state.all_model_checkpoint_paths[
                    -num_last_checkpoints:]

        checkpoints = [c for c in checkpoints if checkpoint_exists(c)]
        if not checkpoints:
            if input_checkpoints:
                raise ValueError(
                        "None of the provided checkpoints exist. %s" % input_checkpoints)
            else:
                raise ValueError("Could not find checkpoints at %s" %
                                                 os.path.dirname(prefix))

        # Read variables from all checkpoints and average them.
        tf.logging.info("Reading variables and averaging checkpoints:")
        for c in checkpoints:
            tf.logging.info("%s ", c)
        var_list = tf.train.list_variables(checkpoints[0])
        var_values, var_dtypes = {}, {}
        for (name, shape) in var_list:
            if not name.startswith("global_step"):
                var_values[name] = np.zeros(shape)
        for checkpoint in checkpoints:
            reader = tf.train.load_checkpoint(checkpoint)
            for name in var_values:
                tensor = reader.get_tensor(name)
                var_dtypes[name] = tensor.dtype
                var_values[name] += tensor
            tf.logging.info("Read from checkpoint %s", checkpoint)
        for name in var_values:    # Average.
            var_values[name] /= len(checkpoints)

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            tf_vars = [
                    tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[v])
                    for v in var_values
            ]
        placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
        assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
        global_step = tf.Variable(
                self.global_step, name="global_step", trainable=False, dtype=tf.int64)
        saver = tf.train.Saver(tf.all_variables())

        # Build a model consisting only of variables, set them to the average values.
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                        six.iteritems(var_values)):
                sess.run(assign_op, {p: value})
            # Use the built saver to save the averaged checkpoint.
            saver.save(sess, output_path, global_step=global_step)

        tf.logging.info("Averaged checkpoints saved in %s", output_path)

    def average_and_zip_model(self, model_folders, saved_model_name):

        def find_model_in_folder(folder):
            file = glob('{}/*.index'.format(folder))
            return file[0].split('.index')[0]

        def zip_model(model_name, output_model):
            # get all ckpt files for latest model
            models = glob('{}*'.format(model_name))
            # zip models into new folder
            z = zipfile.ZipFile(output_model, 'w')
            for model in models:
                z.write(model, compress_type=zipfile.ZIP_DEFLATED)
            z.write('args.txt', compress_type=zipfile.ZIP_DEFLATED)
            z.close()

            # clean up
            for model in models:
                try: os.remove(model)
                except: pass
            try: os.remove('checkpoint')
            except: pass

        models = []
        for folder in model_folders:
            models.append(find_model_in_folder(folder))
        os.rename('{}/args.txt'.format(folder), 'args.txt')
        # average models
        print('\taveraging model parameters...')
        self.avg_checkpoints(input_checkpoints=models, output_path='{}.ckpt'.format(saved_model_name))

        # zip models
        zip_model_name = '{}.zip'.format(saved_model_name)
        zip_model('{}.ckpt'.format(saved_model_name), zip_model_name)
        os.rename('args.txt', '{}/args.txt'.format(folder))
        return zip_model_name



# run code
fed = federated()
fed.federated_training()
