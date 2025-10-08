#include <linux/rbtree.h>
#include "main.h"

void initialize_inputs_db(void) {
	executor.inputs_root = RB_ROOT;
	executor.number_of_inputs = 0;
}
EXPORT_SYMBOL(initialize_inputs_db);

int64_t allocate_input(void) {
	static int64_t input_id = 0;

	struct input_node* new_node = NULL;
	struct rb_node **link = &(executor.inputs_root.rb_node);
	struct rb_node *parent = NULL;

	new_node = (struct input_node*)vzalloc(sizeof(*new_node));

	if(NULL == new_node) {
		module_err("Not enough memory to vmalloc input inside kernel!\n");
		return -ENOMEM;
	}

	RB_CLEAR_NODE(&new_node->node);

	new_node->id = input_id;
	initialize_measurement(&new_node->measurement);

	/* If aux allocation failed inside initialize_measurement, clean up */
	if (new_node->measurement.aux_buffer == NULL) {
		module_err("aux_buffer_alloc failed for new input node, cleaning up\n");
		vfree(new_node);
		return -ENOMEM;
	}

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
EXPORT_SYMBOL(allocate_input);

static struct input_node* get_input_node(int64_t id) {
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

measurement_t* get_measurement(int64_t id) {
	struct input_node* node = get_input_node(id);

	if(NULL == node) return NULL;

	return &(node->measurement);
}
EXPORT_SYMBOL(get_measurement);

input_t* get_input(int64_t id) {
	struct input_node* node = get_input_node(id);

	if (NULL == node) return NULL;

	return &(node->input);
}
EXPORT_SYMBOL(get_input);

void remove_input(int64_t id) {
	struct input_node* node_to_remove = get_input_node(id);
	if(NULL == node_to_remove) return;
	free_measurement(&node_to_remove->measurement);
	rb_erase(&(node_to_remove->node), &(executor.inputs_root));
	vfree(node_to_remove);
	--executor.number_of_inputs;
}
EXPORT_SYMBOL(remove_input);

void destroy_inputs_db(void) {
	struct input_node* node = NULL;
	struct rb_node* rb_node = rb_first(&(executor.inputs_root));

	while(rb_node) {
		node = rb_entry(rb_node, struct input_node, node);
		rb_node = rb_next(rb_node);
		free_measurement(&node->measurement);
		rb_erase(&node->node, &executor.inputs_root);
		vfree(node);
	}

	executor.inputs_root.rb_node = NULL;
	executor.number_of_inputs = 0;
}
EXPORT_SYMBOL(destroy_inputs_db);
