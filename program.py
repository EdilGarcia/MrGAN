import os
from utils import *
from utils_madgan import get_trainable_params
#from networks_madgan import *
from networks_dcgan import *

HEIGHT, WIDTH, CHANNEL = 128, 128, 3
EPOCH = 500
LEARNING_RATE = 2e-4
BATCH_SIZE = 50
NUM_NETS = 3

z_dim = 128
K = 32
G_ITERS = 3
D_ITERS = 5

training_set = 'data_batch_2'
newImg_path = './training_image_output/' + training_set	#Output images path

txt_box = None
tk = None

def set_folders(path):
	global training_set
	valid = 1
	path = path.split('/')
	current = os.getcwd().split('\\')
	curr_len = len(current)
	for i in range(len(current)):
		if current[i] != path[i]:
			valid = 0
			break
	if valid == 1:
		training_set = '/'
		for s in path[curr_len:len(path)]:
			training_set = training_set.join(s+'/')
		print(training_set)

	return valid

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
	g_mmr_new = [0 for i in range(NUM_NETS)]
	d_mmr_new = [0 for i in range(NUM_NETS)]
	log(''.join(str(e)+' ' for e in g_losses))
	for i in range(NUM_NETS):
		for j in range(G_ITERS):
			result = g_losses[i][pairing[j][1][0]]
			#Discriminator win
			if result <= 0.45:
				s1 = 0
				s2 = 1
			#Draw
			elif (result > 0.45) and (result < 0.55):
				s1 = 0.5
				s2 = 0.5
			#Generator win
			elif result >= 0.55:
				s1 = 1
				s2 = 0
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
		if g_mmr_new[pairing[i][1][0]] == 0:
			g_mmr_new[pairing[i][1][0]] = 1500

		d_mmr_new[pairing[i][1][1]] = int(pairing[i][0][1])
		if d_mmr_new[pairing[i][1][1]] == 0:
			d_mmr_new[pairing[i][1][1]] = 1500


	return g_mmr_new, d_mmr_new


def pairings(g_mmr, d_mmr):
	pair = []
	g_mmr_new = []
	d_mmr_new = []
	g_highest, d_highest = [-1, 0], [-1, 0];

	#append flag for visited index
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

	filepath = os.path.join(os.getcwd(),'training_set/'+training_set)
	images = []

	#~get images in folder
	for each in os.listdir(filepath):
		images.append(os.path.join(filepath,each))

	tensorImage = tf.convert_to_tensor(images, dtype = tf.string)
	images_queue = tf.train.slice_input_producer([tensorImage])
	content = tf.read_file(images_queue[0])

	image = tf.image.decode_jpeg(content, channels = CHANNEL)
	image = tf.image.random_flip_left_right(image)
	image = tf.image.random_brightness(image, max_delta = 0.1)
	image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)

	#~resize
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
	g_nets = MRGAN_GEN()
	d_nets = MRGAN_DIS()

	fake_image = [0 for i in range(NUM_NETS)]
	fake_result = [0 for i in range(NUM_NETS)]
	real_result = [0 for i in range(NUM_NETS)]
	fake_image = [0 for i in range(NUM_NETS)]
	d_loss = [0 for i in range(NUM_NETS)]
	g_loss = [0 for i in range(NUM_NETS)]
	d_solver = [0 for i in range(NUM_NETS)]
	g_solver = [0 for i in range(NUM_NETS)]
	dl = [0 for i in range(NUM_NETS)]
	gl = [0 for i in range(NUM_NETS)]
	sig_gloss = []

	#intialize MMR
	g_mmr=[1500 for i in range(NUM_NETS)]
	d_mmr=[1500 for i in range(NUM_NETS)]

	#Initialize Pairing
	pairing = pairings(g_mmr, d_mmr)

	#GAN inputs
	with tf.variable_scope('input'):
		random_input = tf.placeholder(tf.float32, shape=[None, z_dim])	#Latent variable / Random Noise
		real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
		is_train = tf.placeholder(tf.bool, name='is_train')

	#Generators
	for i in range(NUM_NETS):
		common_out_G = g_nets.common_layer_G(random_input, is_train, reuse=True if i > 0 else False, z_dim=z_dim)
		fake_image[i] = g_nets.output_layer_G(common_out_G, is_train, CHANNEL, "generator{}".format(i))

	#Initializers
	pairing_fake, fake_result_loss = get_pairing_fake(pairing)

	#Discriminators
	for i in range(NUM_NETS):
		crd = d_nets.common_layer_D(real_image, is_train, reuse=True if i > 0 else False)
		cfd = d_nets.common_layer_D(fake_image[pairing_fake[i]], is_train, reuse=True)
		real_result[i] = d_nets.output_layer_D(crd, is_train, "discriminator{}".format(i),reuse=False)
		fake_result[i] = d_nets.output_layer_D(cfd, is_train, "discriminator{}".format(i),reuse=True)

	for i in range(NUM_NETS):
		#Loss Functions
		d_loss[i] = tf.reduce_mean(fake_result[fake_result_loss[i]]) - tf.reduce_mean(real_result[i])
		g_loss[i] = -tf.reduce_mean(fake_result[fake_result_loss[i]])
		#Optimizers
		d_solver[i] = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.5) \
		.minimize(d_loss[i], var_list=get_trainable_params('discriminator'))
		g_solver[i] = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.5) \
		.minimize(g_loss[i], var_list=get_trainable_params('generator'))

	#get and convert image to tensor
	batch_size = BATCH_SIZE
	image_batch, num_images = data_preprocess()
	batch_num = int(num_images / batch_size)
	total_batch = 0

	sess = tf.Session()
	tf.reset_default_graph
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	#Tensorboard
	summaries = tf.summary.merge_all()
	writer = tf.summary.FileWriter('./training_summary/'+training_set)
	writer.add_graph(sess.graph)

	#Training Models
	if not os.path.exists('./training_model/' + training_set):
		os.makedirs('./training_model/' + training_set)
		save_path = saver.save(sess, "./training_model/"+training_set+"/model.ckpt")
		ckpt = tf.train.latest_checkpoint('./training_model/' + training_set)
		saver.restore(sess, save_path)
	else:
		ckpt = tf.train.latest_checkpoint('./training_model/' + training_set)
		saver.restore(sess,ckpt)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	log('Total training sample num: {}'.format(num_images))
	log('Batch size: {}, Batch num per epoch: {}, Epoch num: {}'.format(batch_size, batch_num, EPOCH))
	log('tart training...')

	for i in range(EPOCH):
		log('Epoch: {}'.format(i))

		for j in range(batch_num):
			log('Batch: {}'.format(j))
			z = np.random.uniform(-1.0, 1.0, size=[batch_size, z_dim]).astype(np.float32)

			#Train Disciminators
			for k in range(D_ITERS):
				train_image = sess.run(image_batch)
				log('Discriminators {}:'.format(k))
				for l in range(NUM_NETS):
					_,dl[l] = sess.run([d_solver[l], d_loss[l]],feed_dict={random_input: z, real_image: train_image, is_train: True})
					log('{}: loss: {}, act: {},'.format(k, dl[l], get_sigmoid(dl[l])))
			log('\n')
			#Train Generators
			for k in range(G_ITERS):
				log('Generators {}:'.format(k))
				for l in range(NUM_NETS):
					_,gl[l] = sess.run([g_solver[l], g_loss[l]],feed_dict={random_input: z, real_image: train_image, is_train: True})
					sig = get_sigmoid(gl[l])
					log('{}: loss: {}, act: {},'.format(k, gl[l], sig))
				sig_gloss.append([e for e in gl])

			#Update MMR
			g_mmr, d_mmr = update_mmr(pairing, sig_gloss)
			sig_gloss = []

			#Create new pairing
			pairing = pairings(g_mmr, d_mmr)
			pairing_fake, fake_result_loss = get_pairing_fake(pairing)

			log('Pairings:')
			log(''.join(str(e)+' ' for e in pairing))
			log('Pair for Discriminator:')
			log(''.join(str(e)+' ' for e in pairing_fake))
			log('Pair for Generator:')
			log(''.join(str(e)+' ' for e in fake_result_loss))
			log('Ratings:')
			log(''.join(str(e)+' ' for e in g_mmr))
			log(''.join(str(e)+' ' for e in d_mmr))

		if i%10 == 0:
			if not os.path.exists('./training_model/' + training_set):
				os.makedirs('./training_model/' + training_set)
			saver.save(sess, './training_model/' +training_set + '/' + str(i))

			cur_summary = sess.run(summaries, feed_dict={random_input: sample_noise, is_train: False})
			writer.add_summary(cur_summary, i)

		if i%20 == 0:
			# save images
			if not os.path.exists(newImg_path):
				os.makedirs(newImg_path)
			sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, z_dim]).astype(np.float32)
			for k in range(NUM_NETS):
				imgtest = sess.run(fake_image[k], feed_dict={random_input: sample_noise, is_train: False})
				save_images(imgtest, [8,8] ,newImg_path + '/epoch' + str(i) + '_' + str(k) + '.jpg')
		
	coord.request_stop()
	coord.join(threads)
	sess.close()
	print('\nTraining execution stopped')

def generate():
	with tf.variable_scope('input'):
		random_input = tf.placeholder(tf.float32, shape=[None, z_dim])	#Latent variable / Random Noise
		real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
		is_train = tf.placeholder(tf.bool, name='is_train')

	for i in range(NUM_NETS):
		common_out_G = g_nets.common_layer_G(random_input, is_train, reuse=True if i > 0 else False, z_dim=z_dim)
		fake_image[i] = g_nets.output_layer_G(common_out_G, is_train, CHANNEL,"generator{}".format(i))

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	variables_to_restore = slim.get_variables_to_restore(include=['generator'])
	saver = tf.train.Saver(variables_to_restore)
	ckpt = tf.train.latest_checkpoint('./training_model/' + training_set)
	saver.restore(sess, ckpt)

	sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, z_dim]).astype(np.float32)
	for k in range(NUM_NETS):
		imgtest = sess.run(fake_image[k], feed_dict={random_input: sample_noise, is_train: False})
		save_images(imgtest, [8,8] ,newImg_path + '/TEST' +  '_' + str(k) + '.jpg')

	coord.request_stop()
	coord.join(threads)
	sess.close()

if __name__ == "__main__":
	train_mrgan()
