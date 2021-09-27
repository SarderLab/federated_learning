import requests, json, time, os, zipfile, io
import girder_client
from datetime import datetime
from joblib import Parallel, delayed
from glob import glob
from avg_checkpoints import avg_checkpoints

class federated(object):
    """
    A class that provides functions for federated learning of semantic
    segmentaiton of WSIs via multiple distributed servers.
    This uses the Digital slide archive and HistomicsTK-DeepLab to perform training
    """

    def __init__(self):

        ### starting params ###
        self.training_steps_per_round = 1000
        self.training_rounds = 40
        self.global_step = 0
        self.slow_start_step = 750
        self.patch_size = 512
        self.learning_rate = 0.007
        self.learning_rate_start = 0.0001
        self.end_learning_rate = 0.0
        self.learning_power = 0.9
        self.wsi_downsample = [1,2,3,4]
        # self.batch_size = 6
        self.fine_tune_batch_norm = False
        self.augment = 0.0
        self.init_last_layer = False
        self.gpu = "0,1"
        self.num_clones = 2
        self.classes = '["gloms"]'
        self.ignore_label = 'ignore'
        self.output_model = 'IFTA-model.zip'
        self.starting_model = 'model-ImageNet-Xception.zip'
        self.last_layer_gradient_multiplier = 10
        self.last_layers_contain_logits_only = False
        self.upsample_logits = True

        ### server params ###
        self.server_params = []

        ### SERVER 0 ###
        self.server_params.append({
                'token'                 : 'Girder API token',
                'url'                   : 'URL of DSA for server 0',
                'inputFolder'           : 'DSA folder ID for training WSIs',
                'output_model_folder'   : 'DSA folder ID for saved network models',
                'batch_size'            : 12,
                })

        ### SERVER 1 ###
        self.server_params.append({
                'token'                 : 'Girder API token',
                'url'                   : 'URL of DSA for server 1',
                'inputFolder'           : 'DSA folder ID for training WSIs',
                'output_model_folder'   : 'DSA folder ID for saved network models',
                'batch_size'            : 4,
                })

        ### SERVER 2 ###
        self.server_params.append({
                'token'                 : 'Girder API token',
                'url'                   : 'URL of DSA for server 2',
                'inputFolder'           : 'DSA folder ID for training WSIs',
                'output_model_folder'   : 'DSA folder ID for saved network models',
                'batch_size'            : 4,
                })


    ############################################################################
    ### main functions #######################################################
    ############################################################################

    def federated_training(self):
        self.total_training_steps = self.training_steps_per_round * self.training_rounds + self.global_step
        self.end_learning_rate_final = self.end_learning_rate
        self.learning_rate_inital = self.learning_rate

        def training_round(server_dict, starting_model):
            # upload starting model
            inputModelFile = self.upload_model(server_dict, starting_model)
            # train and download model
            model_folder = self.training_cycle(server_dict, inputModelFile)
            return model_folder

        # federated training step
        for round in range(self.training_rounds):
            self.round = round + 1
            print('\nstarting training round {}\n'.format(self.round))
            print('initialize last layer: {}'.format(self.init_last_layer))
            self.training_steps = self.global_step + self.training_steps_per_round

            if self.init_last_layer:
                # run training in parallel
                model_folders = Parallel(n_jobs=len(self.server_params))(delayed(training_round)(server_dict, self.starting_model) for server_dict in self.server_params)
            else:
                # only train one model for round 1
                model_folders = Parallel(n_jobs=1)(delayed(training_round)(server_dict, self.starting_model) for server_dict in self.server_params[:1])

            # use last layer after first training round
            self.init_last_layer = True

            # update global step
            self.global_step = self.training_steps

            # average models
            saved_model_name = 'round-{}-averaged-model'.format(self.round)
            self.starting_model = self.average_and_zip_model(model_folders, saved_model_name)

        # upload final models
        model_folders = Parallel(n_jobs=len(self.server_params))(delayed(self.upload_model)(server_dict, self.starting_model) for server_dict in self.server_params)
        print('\n\nTraining done!\nFinal model created: {}'.format(self.starting_model))



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
        output_model_name = 'round-{}-{}-{}{}'.format(self.round, base, now, ext)

        params = (
            ('inputFolder', args['inputFolder']),
            ('output_model_folder', args['output_model_folder']),
            ('output_model', output_model_name),
            ('inputModelFile', inputModelFile),
            ('WSI_downsample', str(self.wsi_downsample)),
            ('augment', str(self.augment)),
            ('batch_norm', str(self.fine_tune_batch_norm)),
            ('batch_size', str(args['batch_size'])),
            ('classes', self.classes),
            ('ignore_label', str(self.ignore_label)),
            ('init_last_layer', str(self.init_last_layer)),
            ('learning_rate', str(self.learning_rate)),
            ('learning_rate_start', str(self.learning_rate_start)),
            ('num_clones', str(self.num_clones)),
            ('GPU', self.gpu),
            ('patch_size', str(self.patch_size)),
            ('slow_start_step', str(self.slow_start_step)),
            ('steps', str(self.training_steps)),
            ('global_step', str(self.global_step)),
            ('decay_steps', str(self.total_training_steps)),
            ('end_learning_rate', str(self.end_learning_rate)),
            ('learning_power', str(self.learning_power)),
            ('use_xml', 'false'),
            ('last_layer_gradient_multiplier', str(self.last_layer_gradient_multiplier)),
            ('last_layers_contain_logits_only', str(self.last_layers_contain_logits_only)),
            ('upsample_logits', str(self.upsample_logits)),
            ('girderApiUrl', '{}/api/v1'.format(args['url'])),
        )

        response = requests.post('{}/api/v1/slicer_cli_web/brendonl_histomicstk-deeplab_GPU-accelerated-tf/TrainNetwork/run'.format(args['url']), headers=headers, params=params)

        server = args['url'].split('://')[-1]
        server = server.split('.')[0]

        body = json.loads(response.content)
        job_id = body['_id']
        print('\tjob submited on {} with id: {}\n'.format(server, job_id))

        ### check training status ###
        status = 0
        while status <= 2:
            time.sleep(10) # check job status every 10 seconds
            response = requests.get('{}/api/v1/job/{}'.format(args['url'], job_id), headers=headers)
            body = json.loads(response.content)
            status = body['status']
            print('\t{} | {} still training...'.format(datetime.now().strftime("%H:%M:%S"), server), end='\r')

        print('\n\n\t{} done with status code: {}\n\n'.format(server, status))
        try:
            assert status==3, '\n\n!!! training job failed on {} !!!\n\n'.format(args['url'])
        except:
            return None


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
        model_path = '{}-{}'.format(server,model_path)
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        z = zipfile.ZipFile(io.BytesIO(response.content))
        for name in z.namelist():
            z.extract(name, model_path)

        # delete model
        response = requests.delete('{}/api/v1/item/{}'.format(args['url'], model_id), headers=headers)

        return model_path


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
            if folder is not None:
                models.append(find_model_in_folder(folder))
        os.rename('{}/args.txt'.format(folder), 'args.txt')

        if len(models) > 1:
            # average models
            print('\taveraging model parameters...')
            avg_checkpoints(input_checkpoints=models, output_path='{}.ckpt'.format(saved_model_name), global_step_start=self.global_step)
            saved_model_path = saved_model_name
        else:
            saved_model_path = os.path.splitext(models[0])[0]
            models = glob('{}*'.format(saved_model_path))
            for model in models:
                os.rename(model, os.path.basename(model))
            saved_model_path = os.path.basename(saved_model_path)

        # zip models
        zip_model_name = '{}.zip'.format(saved_model_name)
        zip_model('{}.ckpt'.format(saved_model_path), zip_model_name)
        os.rename('args.txt', '{}/args.txt'.format(folder))
        return zip_model_name



# # run code
# fed = federated()
# fed.federated_training()
