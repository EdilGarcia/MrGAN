import os
import random
import math
from utils import *
import sys
sys.path.append("./networks")

HEIGHT, WIDTH, CHANNEL = 128, 128, 3
EPOCH = 500
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_NETS = 3

z_dim = 128
K = 32
G_ITERS = 3
D_ITERS = 5
MODEL_ITERS = 5
SUMMARY_ITERS = 5

NUM_RAND_AVE = 5 #number of times randomizing and will average it
NUM_RAND_THRSH = 0.9 #Percentage of having random

NETWORKS = 1
if NETWORKS == 1:
	from networks_dcgan import *
if NETWORKS == 2:
	from networks_celeba import *

training_set = 'pokemonSet'
training_path ='./training_set'
new_img_path = './output/training_image_output/' + training_set	#Output images path
model_path = './output/training_model/'
summary_path = './output/training_summary/'

txt_box = None
tk = None

def set_folders(path):
	global training_set
	global training_path
	training_path = path
	path = path.split('/')
	current = os.getcwd().split('\\')
	training_set = current[len(current)-1]
	log(training_set)
	log(training_path)

def init_objects(text_box, tki):
	global txt_box
	global tk
	txt_box = text_box
	tk = tki

def log(string):
	if txt_box is not None:
		txt_box.insert(tk.INSERT,string+'\n')
	else:
		print(string)

def get_sigmoid(x):
	return (1 / (1 + np.exp(-x)))

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
			g_mmr_new[pairing[i][1][0]] = 500

		d_mmr_new[pairing[i][1][1]] = int(pairing[i][0][1])
		if d_mmr_new[pairing[i][1][1]] <= 0:
			d_mmr_new[pairing[i][1][1]] = 500

	g_ave = (sum(g_mmr_new))/NUM_NETS
	d_ave = (sum(d_mmr_new))/NUM_NETS
	diff = g_ave - d_ave

	if diff <= -1000:
		D_ITERS = 2
		NUM_RAND_THRSH = 0.25
	elif -750 <= diff <= -999:
		D_ITERS = 3
		NUM_RAND_THRSH = 0.5
	elif -500 <= diff <= -749:
		D_ITERS = 4
		NUM_RAND_THRSH = 0.75
	else:
		D_ITERS = 5
		NUM_RAND_THRSH = 0.9
	log('Rand Update: {} D_ITERS Update: {}'.format(NUM_RAND_THRSH, D_ITERS))
	log('Average G: {} D: {} diff: {}'.format(g_ave, d_ave, diff))

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

			#get highest in D
			for j in range(NUM_NETS):
				if d_mmr_new[j][0] > d_highest[0] and d_mmr_new[j][1] == 0:
					d_highest[0] = d_mmr_new[j][0]
					d_highest[1] = j

			#Set visited indices and save pair
			pair.append([[g_highest[0], d_highest[0]],
						[g_highest[1], d_highest[1]]])
			g_mmr_new[g_highest[1]][1] = 1
			d_mmr_new[d_highest[1]][1] = 1

			g_highest[0] = -1
			d_highest[0] = -1
	return pair

def data_preprocess():
	file_path = training_path+'/'+training_set
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
	g_nets = MRGAN_GEN()
	d_nets = MRGAN_DIS()

	#Discriminator Outputs
	fake_result = [None for i in range(NUM_NETS)]
	real_result = [None for i in range(NUM_NETS)]
	#Generator Outputs
	fake_image = [None for i in range(NUM_NETS*2)]
	#Individual output of unshared networks
	crd = [None for i in range(NUM_NETS)]
	cfd = [None for i in range(NUM_NETS)]
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
	#intialize MMR
	# g_mmr=[1500 for i in range(NUM_NETS)]
	# d_mmr=[1500 for i in range(NUM_NETS)]
	g_mmr=[random.randint(600,1000) for i in range(NUM_NETS)]
	d_mmr=[random.randint(1500,2000) for i in range(NUM_NETS)]
	mutiplier =tf.ones([1,128,128,3])

	# GAN inputs
	with tf.variable_scope('input'):
		random_input = tf.placeholder(tf.float32, shape=[None, z_dim])	#Latent variable / Random Noise
		real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
		is_train = tf.placeholder(tf.bool, name='is_train')
		match = tf.placeholder(tf.int32, name='match')
		index = tf.placeholder(tf.int32, name='match')

	# Initialize Pairings
	pairing = pairings(g_mmr, d_mmr)
	pairing_fake, fake_result_loss = get_pairing_fake(pairing)

	#Generators
	for i in range(NUM_NETS):
		common_out_G = g_nets.shared_G(random_input, is_train, reuse=True if i > 0 else False, z_dim=z_dim)
		fake_image[i] = g_nets.unshared_G(common_out_G, is_train, CHANNEL, "generator{}".format(i))

	# Discriminators
	with tf.control_dependencies(fake_image):
		for i in range(NUM_NETS):
			crd[i] = d_nets.unshared_D(real_image, is_train, "discriminator{}".format(i), reuse=False)
			cfd[i] = d_nets.unshared_D(fake_image[i], is_train, "discriminator{}".format(i), reuse=True)
			real_result[i] = d_nets.shared_D(crd[i], is_train, reuse=True if i > 0 else False)
			fake_result[i] = d_nets.shared_D(cfd[i], is_train, reuse=True)

		for i in range(NUM_NETS):
			#Loss Functions
			d_loss[i] = tf.reduce_mean(fake_result[fake_result_loss[i]]) - tf.reduce_mean(real_result[i])
			g_loss[i] = -tf.reduce_mean(fake_result[fake_result_loss[i]])
			#Optimizers
			d_solver[i] = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.5) \
			.minimize(d_loss[i], var_list=get_trainable_params('discriminator'))
			g_solver[i] = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.5) \
			.minimize(g_loss[i], var_list=get_trainable_params('generator'))

	# Get and convert image to tensor
	batch_size = BATCH_SIZE
	image_batch, num_images = data_preprocess()
	batch_num = int(num_images / batch_size)
	total_batch = 0

	sess = tf.Session()
	tf.reset_default_graph
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	# Tensorboard
	summaries = tf.summary.merge_all()
	writer = tf.summary.FileWriter('./'+summary_path+'/'+training_set)
	writer.add_graph(sess.graph)

	# Training Models
	# if not os.path.exists(model_path+'/'+ training_set):
	# 	os.makedirs(model_path+'/'+ training_set)
	# 	save_path = saver.save(sess, model_path+'/'+training_set+"/model.ckpt")
	# 	ckpt = tf.train.latest_checkpoint(model_path+'/'+ training_set)
	# 	saver.restore(sess, save_path)
	# else:
	# 	ckpt = tf.train.latest_checkpoint(model_path+'/'+training_set)
	# 	saver.restore(sess,ckpt)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	log('Total training sample num: {}'.format(num_images))
	log('Batch size: {}, Batch num per epoch: {}, Epoch num: {}'.format(batch_size, batch_num, EPOCH))
	log('Start training...')

	for i in range(EPOCH):
		for j in range(batch_num):
			log('Epoch: {} Batch: {}'.format(i,j))
			z = np.random.uniform(-1.0, 1.0, size=[batch_size, z_dim]).astype(np.float32)

			# Train Disciminators
			for k in range(D_ITERS):
				train_image = sess.run(image_batch)
				log('Discriminators {}:'.format(k))
				for l in range(NUM_NETS):
					_,dl[l] = sess.run([d_solver[l], d_loss[l]],feed_dict={random_input: z, real_image: train_image, is_train: True})
					log('{}: loss: {}, act: {},'.format(k, dl[l], get_sigmoid(dl[l])))
			# Train Generators
			for k in range(G_ITERS):
				log('Generators {}:'.format(k))
				for l in range(NUM_NETS):
					_,gl[l] = sess.run([g_solver[l], g_loss[l]],feed_dict={random_input: z, real_image: train_image, is_train: True})
					sig = get_sigmoid(gl[l])
					log('{}: loss: {}, act: {},'.format(k, gl[l], sig))
				sig_gloss.append([get_sigmoid(e) for e in gl])

			# Update MMR
			g_mmr, d_mmr = update_mmr(pairing, sig_gloss)
			sig_gloss = []
			# Create new pairing
			pairing = pairings(g_mmr, d_mmr)
			pairing_fake, fake_result_loss = get_pairing_fake(pairing)

			log('Pairings:')
			log('{}'.format(pairing))
			log('Pair for Discriminator:')
			log('{}'.format(pairing_fake))
			log('Pair for Generator:')
			log('{}'.format(fake_result_loss))
			log('Ratings:')
			log('{}'.format(g_mmr))
			log('{}'.format(d_mmr))

		if i%MODEL_ITERS == 0:
			if not os.path.exists(model_path+'/'+training_set):
				os.makedirs(model_path+'/'+training_set)
			saver.save(sess,model_path+'/'+training_set +'/'+str(i))

			cur_summary = sess.run(summaries, feed_dict={random_input: z, is_train: False})
			writer.add_summary(cur_summary, i)

		if i%SUMMARY_ITERS == 0:
			# save images
			if not os.path.exists(new_img_path):
				os.makedirs(new_img_path)
			sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, z_dim]).astype(np.float32)
			for k in range(NUM_NETS):
				imgtest = sess.run(fake_image[k], feed_dict={random_input: sample_noise, is_train: False})
				out_size = int(math.sqrt(BATCH_SIZE))
				out_size2 = out_size
				if (out_size * out_size) != BATCH_SIZE:
					out_size2 = out_size2+1
				save_images(imgtest, [out_size,out_size2] ,new_img_path + '/epoch' + str(i) + '_' + str(k) + '.jpg')

	coord.request_stop()
	coord.join(threads)
	sess.close()
	print('\nTraining execution stopped')
	exit()
def generate():
	g_nets = MRGAN_GEN()
	fake_image = [None for i in range(NUM_NETS)]
	with tf.variable_scope('input'):
		random_input = tf.placeholder(tf.float32, shape=[None, z_dim])	#Latent variable / Random Noise
		real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
		is_train = tf.placeholder(tf.bool, name='is_train')

	for i in range(NUM_NETS):
		common_out_G = g_nets.shared_G(random_input, is_train, reuse=True if i > 0 else False, z_dim=z_dim)
		fake_image[i] = g_nets.unshared_G(common_out_G, is_train, CHANNEL, "generator{}".format(i))

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	variables_to_restore = slim.get_variables_to_restore(include=['generator'])
	saver = tf.train.Saver(variables_to_restore)
	ckpt = tf.train.latest_checkpoint(model_path+training_set)
	saver.restore(sess, ckpt)

	sample_noise = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, z_dim]).astype(np.float32)
	for k in range(NUM_NETS):
		imgtest = sess.run(fake_image[k], feed_dict={random_input: sample_noise, is_train: False})
		out_size = int(math.sqrt(BATCH_SIZE))
		out_size2 = out_size
		if (out_size * out_size) != BATCH_SIZE:
			out_size2 = out_size2+1
		save_images(imgtest, [out_size,out_size2] ,'./training_image_output/TEST_'+str(k)+'.jpg')
	coord.request_stop()
	coord.join(threads)
	sess.close()

if __name__ == "__main__":
	train_mrgan()
	# generate()
