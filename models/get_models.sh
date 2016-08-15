#!/bin/bash

function get_alexnet {
	if [[ -f bvlc_alexnet.npy ]];
	then
		echo "bvlc_alexnet.npy already exists."
	else
		wget http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy
	fi
}

function get_vgg {
	if [[ -f vgg16_weights.npz ]];
	then
		echo "vgg16_weights.npz already exists."
	else
		wget http://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz
	fi
}

while test $# -gt 0; do
    case "$1" in
            -h|--help)
                    echo "Download pre-trained models"
                    echo " "
                    echo "get_models.sh [arguments]"
                    echo " "
                    echo "options:"
                    echo "-h, --help                show brief help"
                    echo "-m       					specify a model to download"
                    exit 0
                    ;;
            -m)
                    shift
                    if test $# -gt 0; then
                    	if [ "$1" != alexnet ] && [ "$1" != vgg ]; then
                    		echo "the model should be 'vgg' or 'alexnet'"
                    		exit 1
                    	fi
                        export MODEL=$1
                    else
                        echo "no model specified"
                        exit 1
                    fi
                    shift
                    ;;
            *)
                    break
                    ;;
    esac
done

if [ -z "$MODEL" ]; then
	get_alexnet
	get_vgg
elif [[ "$MODEL" == alexnet ]]; then
	get_alexnet
elif [[ "$MODEL" == vgg ]]; then
	get_vgg
fi

exit 1
