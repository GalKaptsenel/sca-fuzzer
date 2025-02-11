#include <linux/rbtree.h>
#include "main.h"

void initialize_inputs_db(void) {
	executor.inputs_root = RB_ROOT;
	executor.number_of_inputs = 0;
}

int allocate_input(void) {
	static int input_id = 0;

	struct input_node* new_node = NULL;
	struct rb_node **link = &(executor.inputs_root.rb_node);
	struct rb_node *parent = NULL;

	new_node = (struct input_node*)vmalloc(sizeof(struct input_node));

	if(NULL == new_node) {
		module_err("Not enough memory to vmalloc input inside kernel!\n");
		return -ENOMEM;
	}

	new_node->id = input_id;
	memset(&(new_node->input), 0, sizeof(input_t));
	initialize_measurement(&(new_node->measurement));

	while(*link) {
		parent = *link;
		if(new_node->id < rb_entry(parent, struct input_node, node)->id) {
			link = &(parent->rb_left);
		}
		else {
			link = &(parent->rb_right);
		}
	}

	rb_link_node(&(new_node->node), parent, link);
	rb_insert_color(&(new_node->node), &executor.inputs_root);
	++executor.number_of_inputs;
	++input_id;

	return new_node->id;
}

static struct input_node* get_input_node(int id) {
	struct rb_node* node = executor.inputs_root.rb_node;

	while(node) {
		struct input_node* data = rb_entry(node, struct input_node, node);

		if(id < data->id) {
			node = node->rb_left;
		}
		else if (id > data->id) {
			node = node->rb_right;
		}
		else {
			return data;
		}
	}

	return NULL;
}

input_t* get_input(int id) {
	struct input_node* node = get_input_node(id);

	if (NULL == node) return NULL;

	return &(node->input);
}

void remove_input(int id) {
	struct input_node* node_to_remove = get_input_node(id);
	if(NULL == node_to_remove) return;
	rb_erase(&(node_to_remove->node), &(executor.inputs_root));
	vfree(node_to_remove);
	--executor.number_of_inputs;
}

void destroy_inputs_db(void) {
	struct input_node* node = NULL;
	struct rb_node* rb_node = rb_first(&(executor.inputs_root));

	while(rb_node) {
		node = rb_entry(rb_node, struct input_node, node);
		rb_node = rb_next(rb_node);
		vfree(node);
	}

	executor.inputs_root.rb_node = NULL;
	executor.number_of_inputs = 0;
}
