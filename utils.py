from sklearn.datasets import fetch_mldata

def get_MNIST():
    """Returns a (data, target) tuple of the MNIST dataset"""
    mnist = fetch_mldata('MNIST original', data_home='./data/')
    return (mnist.data, mnist.target)