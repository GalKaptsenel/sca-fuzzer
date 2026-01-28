#include <Python.h>
#include <stdio.h>
#include <stdint.h>

static PyObject *tage_instance = NULL;
static int python_initialized = 0;

int python_init() {
    if (!python_initialized) {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            fprintf(stderr, "Failed to initialize Python\n");
            return -1;
        }
        python_initialized = 1;
    }
    return 0;
}

int tagebp_init(const char *module_dir, const char *module_name,
		int num_state_bits, int init_state_val, int num_base_entries) {

	if (!python_initialized) {
		if (python_init() != 0) return -1;
	}

	// If a previous instance exists, release it
	if (tage_instance) {
		Py_DECREF(tage_instance);
		tage_instance = NULL;
	}

	// Add module_dir to sys.path
	PyObject *sys_path = PySys_GetObject("path"); // Borrowed reference
	PyObject *py_path = PyUnicode_FromString(module_dir);
	if (!py_path) {
		PyErr_Print();
		return -1;
	}
	PyList_Append(sys_path, py_path);
	Py_DECREF(py_path);

	// Import the module
	PyObject *py_module_name = PyUnicode_FromString(module_name);
	PyObject *module = PyImport_Import(py_module_name);
	Py_DECREF(py_module_name);
	if (!module) {
		PyErr_Print();
		return -1;
	}

	// Get the TageBP class
	PyObject *tage_class = PyObject_GetAttrString(module, "TageBP");
	Py_DECREF(module);
	if (!tage_class) {
		PyErr_Print();
		return -1;
	}

	// Build constructor arguments and create instance
	PyObject *args = Py_BuildValue("(iii)", num_state_bits, init_state_val, num_base_entries);
	tage_instance = PyObject_CallObject(tage_class, args);
	Py_DECREF(args);
	Py_DECREF(tage_class);

	if (!tage_instance) {
		PyErr_Print();
		return -1;
	}

	return 0;
}

int tagebp_predict(uintptr_t address) {
	if (!tage_instance) return -1;

	PyObject *py_addr = PyLong_FromUnsignedLong(address);
	PyObject *result = PyObject_CallMethod(tage_instance, "predict", "O", py_addr);
	Py_DECREF(py_addr);

	if (!result) {
		PyErr_Print();
		return -1;
	}

	int prediction = PyObject_IsTrue(result);
	Py_DECREF(result);
	return prediction;
}

void tagebp_update(uintptr_t address, int taken) {
	if (!tage_instance) return;

	PyObject *py_addr = PyLong_FromUnsignedLong(address);
	PyObject *py_taken = PyBool_FromLong(taken);
	PyObject *result = PyObject_CallMethod(tage_instance, "update", "OO", py_addr, py_taken);
	Py_DECREF(py_addr);
	Py_DECREF(py_taken);

	if (!result) {
		PyErr_Print();
	} else {
		Py_DECREF(result); // update returns None
	}
}

void tagebp_destroy_instance() {
	fprintf(stderr, "In tagebp destroy instance start\n");
    if (tage_instance) {
	fprintf(stderr, "In tagebp destroy instance tage_isntance DECREF\n");
        Py_DECREF(tage_instance);
	fprintf(stderr, "In tagebp destroy instance tage_isntance DECREF - finished\n");
        tage_instance = NULL;
    }
	fprintf(stderr, "In tagebp destroy instance end\n");
}

void python_finalize() {
    // Destroy any remaining instance
    tagebp_destroy_instance();

    if (python_initialized) {
        // Optional: remove module from sys.modules to reduce hanging
        PyObject *modules = PyImport_GetModuleDict();
        if (modules) {
            PyDict_DelItemString(modules, "bootstrap_director");
        }

	fprintf(stderr, "In python finilize calling Py_FinalizeEx()\n");
        Py_FinalizeEx();
	fprintf(stderr, "In python finilize out of Py_FinalizeEx()\n");
        python_initialized = 0;
    }
}

