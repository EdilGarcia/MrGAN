import os
import random
import math
from utils import *
from tensorflow.python.framework import ops
from eval_funcs import *
from data_synthetic import *
import sys
sys.path.append("./networks")
sys.path.append("./training_set")
import networks_dcgan as dc_net
import networks_celeba as celeb_net
import networks_synthetic as synt_net
plt.ion()

HEIGHT, WIDTH, CHANNEL = 128, 128, 3
EPOCH = 11
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_NETS = 3

z_dim = 128
K = 32
G_ITERS = 3
D_ITERS = 5
MODEL_ITERS = 1
SAVE_ITERS = 1

NUM_RAND_AVE = 5 #number of times randomizing and will average it
NUM_RAND_THRSH = 0.9 #Percentage of having random

NETWORKS = 1

generated = 0

training_set = 'pokemonSet'
training_path ='./training_set/' + training_set
output_path = './output'
model_path = output_path + '/training_model/'
new_img_path = output_path + '/training_image_output/' + training_set	#Output images path
summary_path = output_path + '/training_summary/'

txt_box = None
tk = None

def write_config():
	fh = open(model_path+'/config.txt', "w")
	fh.write('EPOCH='+str(EPOCH))
	fh.write('\nLEARNING_RATE='+str(LEARNING_RATE))
	fh.write('\nBATCH_SIZE='+str(BATCH_SIZE))
	fh.write('\nNUM_NETS='+str(NUM_NETS))
	fh.write('\nMODEL_ITERS='+str(MODEL_ITERS))
	fh.write('\nSAVE_ITERS='+str(SAVE_ITERS))
	fh.write('\nNUM_RAND_AVE='+str(NUM_RAND_AVE))
	fh.write('\nNUM_RAND_THRSH='+str(NUM_RAND_THRSH))
	fh.write('\nNETWORKS='+str(NETWORKS))
	fh.close

def set_params(settings_dict):
	global EPOCH, LEARNING_RATE, BATCH_SIZE, NUM_NETS, MODEL_ITERS, SAVE_ITERS, NUM_RAND_AVE, NUM_RAND_THRSH, NETWORKS
	EPOCH = int(settings_dict['EPOCH'])
	LEARNING_RATE = float(settings_dict['LEARNING_RATE'])
	BATCH_SIZE = int(settings_dict['BATCH_SIZE'])
	NUM_NETS = int(settings_dict['NUM_NETS'])
	MODEL_ITERS = int(settings_dict['MODEL_ITERS'])
	SAVE_ITERS = int(settings_dict['SAVE_ITERS'])
	NUM_RAND_AVE = int(settings_dict['NUM_RAND_AVE'])
	NUM_RAND_THRSH = float(settings_dict['NUM_RAND_THRSH'])
	NETWORKS = int(settings_dict['NETWORKS'])

def set_folders(set_dir, model_dir, out_dir):
	global training_set, training_path, output_path, model_path
	training_path = set_dir
	set_dir = set_dir.split('/')
	training_set = set_dir[len(set_dir)-1]
	log(training_path)
	model_path = model_dir
	if model_dir == '' or model_dir is None:
		model_path = out_dir+'/training_model/'+training_set

def init_objects(text_box, tki):
	global txt_box
	global tk
	txt_box = text_box
	tk = tki

def log(string):
	if txt_box is not None:
		txt_box.insert(tk.INSERT,string+'\n')
	print(string)

def get_pairing_fake(pairing):
	fake = [i for i in range(NUM_NETS)]
	results = [i for i in range(NUM_NETS)]
	for k in range(len(pairing)):
		for i in range(len(pairing)):
			if pairing[i][1][1] == k:
				fake[k] = pairing[i][1][0]
			if(pairing[i][1][0]) == k:
				results[k] = pairing[i][1][1]
	return fake, results
	# return tf.Variable(fake, dtype=tf.int32, name='fake'), tf.Variable(results, dtype=tf.int32, name='results')

def update_mmr(pairing, g_losses):
	global NUM_RAND_THRSH
	global D_ITERS
	g_mmr_new = [0 for i in range(NUM_NETS)]
	d_mmr_new = [0 for i in range(NUM_NETS)]
	log(''.join(str(e)+' ' for e in g_losses))
	for i in range(NUM_NETS):
		for j in range(G_ITERS):
			result = g_losses[i][pairing[j][1][0]]
			#Generator win
			if result <= 0.45:
				s1 = 1
				s2 = 0
			#Draw
			elif (result > 0.45) and (result < 0.55):
				s1 = 0.5
				s2 = 0.5
			#Discriminator win
			elif result >= 0.55:
				s1 = 0
				s2 = 1
			#Calculate Transform Rating
			r1 = 10 ** (pairing[j][0][0] / 400)
			r2 = 10 ** (pairing[j][0][1] / 400)
			#Get Expected Score
			e1 = r1 / (r1 + r2)
			e2 = r2 / (r1 + r2)

			pairing[j][0][0] = pairing[j][0][0] + K * (s1 - e1)
			pairing[j][0][1] = pairing[j][0][1] + K * (s2 - e2)

	for i in range(NUM_NETS):
		g_mmr_new[pairing[i][1][0]] = int(pairing[i][0][0])
		if g_mmr_new[pairing[i][1][0]] <= 0:
			g_mmr_new[pairing[i][1][0]] = 1200

		d_mmr_new[pairing[i][1][1]] = int(pairing[i][0][1])
		if d_mmr_new[pairing[i][1][1]] <= 0:
			d_mmr_new[pairing[i][1][1]] = 1200

	g_ave = (sum(g_mmr_new))/NUM_NETS
	d_ave = (sum(d_mmr_new))/NUM_NETS
	diff = g_ave - d_ave

	if diff <= -1000:
		# D_ITERS = 2
		NUM_RAND_THRSH = 0.25
	elif -750 <= diff <= -999:
		# D_ITERS = 3
		NUM_RAND_THRSH = 0.5
	elif -500 <= diff <= -749:
		# D_ITERS = 4
		NUM_RAND_THRSH = 0.75
	else:
		# D_ITERS = 5
		NUM_RAND_THRSH = 0.9
	log('Rand Update: {} D_ITERS Update: {}'.format(NUM_RAND_THRSH, D_ITERS))
	log('Average G: {:.5} D: {:.5} diff: {:.5}'.format(g_ave, d_ave, diff))

	return g_mmr_new, d_mmr_new

def pairings(g_mmr, d_mmr):
	rand = [random.randint(0,100)/100 for i in range(NUM_RAND_AVE)]
	ave = sum(rand)/NUM_RAND_AVE

	pair = []
	g_mmr_new = []
	d_mmr_new = []
	g_highest, d_highest = [-1, 0], [-1, 0];

	#append flag for visited index
	if NUM_RAND_THRSH <= ave:
		g = [i for i in range(NUM_NETS)]
		d = [i for i in range(NUM_NETS)]
		random.shuffle(g)
		random.shuffle(d)
		for i in range(NUM_NETS):
			pair.append([[g_mmr[g[i]], d_mmr[d[i]]],
						 [g[i], d[i]]])
		log('Shuffled!')
	else:
		for i in range(NUM_NETS):
			g_mmr_new.append([g_mmr[i],0])
			d_mmr_new.append([d_mmr[i],0])
		for i in range(NUM_NETS):
			#get highest in G
			for j in range(NUM_NETS):
				if g_mmr_new[j][0] > g_highest[0] and g_mmr_new[j][1] == 0:
					g_highest[0] = g_mmr_new[j][0]
					g_highest[1] = j

			# get highest in D
			for j in range(NUM_NETS):
				if d_mmr_new[j][0] > d_highest[0] and d_mmr_new[j][1] == 0:
					d_highest[0] = d_mmr_new[j][0]
					d_highest[1] = j

			# Set visited indices and save pair
			pair.append([[g_highest[0], d_highest[0]],
						[g_highest[1], d_highest[1]]])
			g_mmr_new[g_highest[1]][1] = 1
			d_mmr_new[d_highest[1]][1] = 1

			g_highest[0] = -1
			d_highest[0] = -1
	return pair

def data_preprocess():
	file_path = training_path
	images = []
	# Get images in folder
	for each in os.listdir(file_path):
		images.append(os.path.join(file_path,each))

	tensorImage = tf.convert_to_tensor(images, dtype = tf.string)
	images_queue = tf.train.slice_input_producer([tensorImage])
	content = tf.read_file(images_queue[0])
	image = tf.image.decode_jpeg(content, channels = CHANNEL)
	image = tf.image.random_flip_left_right(image)
	image = tf.image.random_brightness(image, max_delta = 0.1)
	image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)

	# Resize
	imageSize = [HEIGHT, WIDTH]
	image = tf.image.resize_images(image, imageSize)
	image.set_shape([HEIGHT,WIDTH,CHANNEL])
	image = tf.cast(image, tf.float32)
	# image = tf.cast(image, tf.uint8)
	image = image / 255.0

	imagesBatch = tf.train.shuffle_batch([image], batch_size = BATCH_SIZE,
					num_threads = 4, capacity = 200 + 3 * BATCH_SIZE,
					min_after_dequeue = 200)

	numImages = len(images)

	return imagesBatch,numImages

def train_mrgan():
	#Variables
	if NETWORKS == 1:
		g_nets = dc_net.MRGAN_GEN()
		d_nets = dc_net.MRGAN_DIS()

	elif NETWORKS == 2:
		g_nets = celeb_net.MRGAN_GEN()
		d_nets = celeb_net.MRGAN_DIS()
	else:
		global training_set
		training_set = 'synthetic'
		g_nets = synt_net.MRGAN_SYNTHETHIC_G(2, z_dim, z_dim*2)
		d_nets = synt_net.MRGAN_SYNTHETHIC_D(1, 2, z_dim*2)

	#Discriminator Outputs
	fake_result = [None for i in range(NUM_NETS)]
	real_result = [None for i in range(NUM_NETS)]
	#Generator Outputs
	fake_image = [None for i in range(NUM_NETS)]
	paired_images = [None for i in range(NUM_NETS)]
	#Individual output of unshared networks
	crd = [None for i in range(NUM_NETS)]
	cfd = [None for i in range(NUM_NETS)]
	fr = [None for i in range(NUM_NETS)]
	#Loss values
	d_loss = [None for i in range(NUM_NETS)]
	g_loss = [None for i in range(NUM_NETS)]
	#Optimizers
	d_solver = [None for i in range(NUM_NETS)]
	g_solver = [None for i in range(NUM_NETS)]
	#Calculated Loss
	dl = [None for i in range(NUM_NETS)]
	gl = [None for i in range(NUM_NETS)]
	#Tanh Loss for pairing results
	sig_gloss = []
	# intialize MMR
	g_mmr=[1500 for i in range(NUM_NETS)]
	d_mmr=[1500 for i in range(NUM_NETS)]

	# Get and convert image to tensor
	batch_size = BATCH_SIZE
	if NETWORKS != 3:
		image_batch, num_images = data_preprocess()
		shape = image_batch.shape
		batch_num = int(num_images / batch_size)
	else:
		data = Spiral()
		shape = data.train.images[0].shape
		shape = (None, ) + shape
		batch_num = batch_size
	# GAN inputs

	with tf.variable_scope('input'):
		random_input = tf.placeholder(tf.float32, shape=[None, z_dim])	#Latent variable / Random Noise
		real_image = tf.placeholder(tf.float32, shape=shape, name='real_image')
		is_train = tf.placeholder(tf.bool, name='is_train')
		var_pairing = tf.placeholder(tf.int32, shape=[3], name='pairing')
		var_fake_res = tf.placeholder(tf.int32, shape=[3], name='fake_results')

	# Initialize Pairings
	pairing = pairings(g_mmr, d_mmr)
	fake, results = get_pairing_fake(pairing)

	# Match making Variables
	pairing_fake = tf.Variable(fake, dtype=tf.int32, name='fake')
	fake_result_loss = tf.Variable(results, dtype=tf.int32, name='results')

	#Assig Op
	pairing_fake_assign = tf.assign(pairing_fake, var_pairing)
	pairing_result_assign = tf.assign(fake_result_loss, var_fake_res)

	# Generators
	for i in range(NUM_NETS):
		common_out_G = g_nets.shared(random_input, is_train, 'generator', reuse=True if i > 0 else False, z_dim=z_dim)
		fake_image[i] = g_nets.unshared(common_out_G, is_train, "generator{}".format(i))

    # Pairing Network
	ret_fake_img_1 = fake_image[0]
	ret_fake_img_2 = fake_image[1]
	ret_fake_img_3 = fake_image[2]
	for i in range(NUM_NETS):
		final = tf.where(tf.equal(pairing_fake[i], 1), x=ret_fake_img_2, y=ret_fake_img_3)
		paired_images[i] = tf.where(tf.equal(pairing_fake[i], 0), x=ret_fake_img_1, y=final)

	# Discriminators
	for i in range(NUM_NETS):
		crd[i] = d_nets.unshared(real_image, is_train, "discriminator{}".format(i), reuse=False)
		cfd[i] = d_nets.unshared(paired_images[i], is_train, "discriminator{}".format(i), reuse=True)
		real_result[i] = d_nets.shared(crd[i], is_train, 'discriminator', reuse=True if i > 0 else False)
		fake_result[i] = d_nets.shared(cfd[i], is_train, 'discriminator', reuse=True)

	# Loss Functions and Optimizers
	ret_fake_res_1 = fake_result[0]
	ret_fake_res_2 = fake_result[1]
	ret_fake_res_3 = fake_result[2]

	for i in range(NUM_NETS):
		final2 = tf.where(tf.equal(fake_result_loss[i], 1), x=ret_fake_res_2, y=ret_fake_res_3)
		fr[i] = tf.where(tf.equal(fake_result_loss[i], 0), x=ret_fake_res_1, y=final2)

		d_loss[i] = tf.reduce_mean(fr[i]) - tf.reduce_mean(real_result[i])
		g_loss[i] = -tf.reduce_mean(fr[i])
		d_solver[i] = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.5) \
			.minimize(d_loss[i], var_list=get_trainable_params('discriminator'))
		g_solver[i] = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.5) \
			.minimize(g_loss[i], var_list=get_trainable_params('generator'))

	sess = tf.Session()
	tf.reset_default_graph
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	# Tensorboard
	summaries = tf.summary.merge_all()
	writer = tf.summary.FileWriter('./'+summary_path+'/'+training_set)
	writer.add_graph(sess.graph)

	Training Models
	if not os.path.exists(model_path):
		os.makedirs(model_path)
		save_path = saver.save(sess, model_path+"/model.ckpt")
		ckpt = tf.train.latest_checkpoint(model_path)
		saver.restore(sess, save_path)
		write_config()
	else:
		ckpt = tf.train.latest_checkpoint(model_path)
		saver.restore(sess,ckpt)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	if NETWORKS != 3:
		log('Total training sample num: {}'.format(num_images))
		log('Batch size: {}, Batch num per epoch: {}, Epoch num: {}'.format(batch_size, batch_num, EPOCH))
	log('Start training...')

	for i in range(EPOCH):
		for j in range(batch_num):
			log('Epoch: {} Batch: {}'.format(i,j))
			# Generate Noise
			z = np.random.uniform(-1.0, 1.0, size=[batch_size, z_dim]).astype(np.float32)

			# Data set batches
			if NETWORKS != 3:
				train_image = sess.run(image_batch)
			else:
				train_image, _ = data.train.next_batch(batch_size)	# Synthethic Dataset

			# Assign new Paring for discriminators and its Gradients
			_, pf = sess.run([pairing_fake_assign, pairing_fake], feed_dict={random_input: z, real_image: train_image, is_train: True, var_pairing: fake, var_fake_res: results})

			# Train Disciminators
			for k in range(D_ITERS):

				log('Discriminators {}:'.format(k))
				for l in range(NUM_NETS):
					_, dl[l] = sess.run([d_solver[l], d_loss[l]],feed_dict={random_input: z, real_image: train_image, is_train: True, var_pairing: fake, var_fake_res: results})
					log('{}: loss: {:.10}, act: {:.10},'.format(k, dl[l], get_sigmoid(dl[l])))

			# Train Generators
			for k in range(G_ITERS):
				log('Generators {}:'.format(k))
				for l in range(NUM_NETS):
					_, gl[l] = sess.run([g_solver[l], g_loss[l]],feed_dict={random_input: z, real_image: train_image, is_train: True, var_pairing: fake, var_fake_res: results})
					sig = get_sigmoid(gl[l])
					log('{}: loss: {:.10}, act: {:.10},'.format(k, gl[l], sig))
				sig_gloss.append([get_sigmoid(e) for e in gl])

			# Update MMR
			g_mmr, d_mmr = update_mmr(pairing, sig_gloss)
			sig_gloss = []
			# Create new pairing
			pairing = pairings(g_mmr, d_mmr)
			fake, results = get_pairing_fake(pairing)

			log('Pairings:')
			log('{}'.format(pairing))
			log('Pair for Discriminator:')
			log('{}'.format(fake))
			log('Pair for Generator:')
			log('{}'.format(results))
			log('Ratings:')
			log('{}'.format(g_mmr))
			log('{}'.format(d_mmr))

		if i%MODEL_ITERS == 0:
			# Save model and summaries
			if not os.path.exists(model_path):
				os.makedirs(model_path)
			saver.save(sess,model_path+'/'+str(i))
			cur_summary = sess.run(summaries, feed_dict={random_input: z, is_train: False, var_pairing: fake, var_fake_res: results})
			writer.add_summary(cur_summary, i)

		if i%SAVE_ITERS == 0:
			# save images
			if not os.path.exists(new_img_path):
				os.makedirs(new_img_path)
			sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, z_dim]).astype(np.float32)
			for k in range(NUM_NETS):
				imgtest = sess.run(fake_image[k], feed_dict={random_input: sample_noise, is_train: False, var_pairing: fake, var_fake_res: results})

				out_size = int(math.sqrt(BATCH_SIZE))
				out_size2 = out_size
				if (out_size * out_size) != BATCH_SIZE:
					out_size2 = out_size2+1
				if NETWORKS != 3:
					save_images(imgtest, [out_size,out_size2] ,new_img_path + '/epoch' + str(i) + '_' + str(k) + '.jpg')
				else:
					plt.scatter(imgtest[:,0], imgtest[:,1], alpha=0.1)
					plt.savefig(new_img_path+'/synt{}'.format(k), bbox_inches='tight')

	coord.join(threads)
	coord.request_stop()
	sess.close()
	print('\nTraining execution stopped')

def generate():
	if NETWORKS == 1:
		g_nets = dc_net.MRGAN_GEN()
	elif NETWORKS == 2:
		g_nets = celeb_net.MRGAN_GEN()
	else:
		global training_set
		training_set = 'synthetic'
		g_nets = synt_net.MRGAN_SYNTHETHIC_G(2, z_dim, z_dim*2)

	ops.reset_default_graph()

	fake_image = [None for i in range(NUM_NETS)]

	with tf.variable_scope('input'):
		random_input = tf.placeholder(tf.float32, shape=[None, z_dim])	#Latent variable / Random Noise
		real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
		is_train = tf.placeholder(tf.bool, name='is_train')

	for i in range(NUM_NETS):
		common_out_G = g_nets.shared(random_input, is_train, 'generator', reuse=True if i > 0 else False, z_dim=z_dim)
		fake_image[i] = g_nets.unshared(common_out_G, is_train, "generator{}".format(i))

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	variables_to_restore = slim.get_variables_to_restore(include=['generator'])
	saver = tf.train.Saver(variables_to_restore)
	ckpt = tf.train.latest_checkpoint(model_path)
	saver.restore(sess, ckpt)

	# Image Generation
	sample_noise = np.random.uniform(-1.0, 1.0, size=[1024, z_dim]).astype(np.float32)
	for k in range(NUM_NETS):
		imgtest = sess.run(fake_image[k], feed_dict={random_input: sample_noise, is_train: False})
		out_size = int(math.sqrt(BATCH_SIZE))
		out_size2 = out_size
		if (out_size * out_size) != BATCH_SIZE:
			out_size2 = out_size2+1
		if NETWORKS != 3:
			new_img_path = output_path + '/training_image_output/' + training_set
			save_images(imgtest, [out_size,out_size2], new_img_path+'/TEST_'+str(k)+'.jpg')
		else:
			new_img_path = output_path + '/training_image_output/' + training_set
			plt.scatter(imgtest[:,0], imgtest[:,1], alpha=0.1)
			plt.savefig(new_img_path+'/TEST_{}'.format(k), bbox_inches='tight')

	# Image Evaluation


	coord.join(threads)
	coord.request_stop()
	sess.close()
	log('Done!')
if __name__ == "__main__":
	train_mrgan()
	# generate()
