#include <Python.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "tage_py.h"   /* keep these definitions in sync with the declarations */

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

int tagebp_init(const char *module_dir, const char *module_name) {

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
	if (!sys_path || !py_path) {
		PyErr_Print();
		Py_XDECREF(py_path);
		return -1;
	}
	if (PyList_Append(sys_path, py_path) != 0) {
		PyErr_Print();
		Py_DECREF(py_path);
		return -1;
	}
	Py_DECREF(py_path);

	// Import the module
	PyObject *py_module_name = PyUnicode_FromString(module_name);
	PyObject *module = PyImport_Import(py_module_name);
	Py_DECREF(py_module_name);
	if (!module) {
		PyErr_Print();
		return -1;
	}

	// Construct the predictor via the module factory, which owns predictor selection and config
	// (e.g. the PHR fold width) — keeping this C layer model-agnostic.
	tage_instance = PyObject_CallMethod(module, "create_predictor", NULL);
	Py_DECREF(module);

	if (!tage_instance) {
		PyErr_Print();
		return -1;
	}

	return 0;
}

int tagebp_predict(uintptr_t address) {
	if (!tage_instance) return -1;

	PyObject *py_addr = PyLong_FromUnsignedLongLong(address);
	PyObject *result = PyObject_CallMethod(tage_instance, "predict", "O", py_addr);
	Py_XDECREF(py_addr);

	if (!result) {
		PyErr_Print();
		return -1;
	}

	int prediction = PyObject_IsTrue(result);   /* -1 on error */
	Py_DECREF(result);
	if (prediction < 0) {
		PyErr_Print();
		return -1;
	}
	return prediction;
}

void tagebp_update(uintptr_t address, int taken, uintptr_t target) {
	if (!tage_instance) return;

	PyObject *py_addr = PyLong_FromUnsignedLongLong(address);
	PyObject *py_taken = PyBool_FromLong(taken);
	PyObject *py_target = PyLong_FromUnsignedLongLong(target);
	PyObject *result = PyObject_CallMethod(tage_instance, "update", "OOO", py_addr, py_taken, py_target);
	Py_XDECREF(py_addr);
	Py_XDECREF(py_taken);
	Py_XDECREF(py_target);

	if (!result) {
		PyErr_Print();
		fprintf(stderr, "[CE FATAL] tagebp_update: TAGE update() raised\n");
		abort();
	}
	Py_DECREF(result); // update returns None
}

void tagebp_advance(uintptr_t address, int taken, uintptr_t target) {
	if (!tage_instance) return;

	PyObject *py_addr = PyLong_FromUnsignedLongLong(address);
	PyObject *py_taken = PyBool_FromLong(taken);
	PyObject *py_target = PyLong_FromUnsignedLongLong(target);
	PyObject *result = PyObject_CallMethod(tage_instance, "advance", "OOO", py_addr, py_taken, py_target);
	Py_XDECREF(py_addr);
	Py_XDECREF(py_taken);
	Py_XDECREF(py_target);

	if (!result) {
		PyErr_Print();
		fprintf(stderr, "[CE FATAL] tagebp_advance: TAGE advance() raised\n");
		abort();
	}
	Py_DECREF(result); // advance returns None
}

/* checkpoint / rollback / commit take no args and return None. A raised exception here is a fatal
 * engine bug (e.g. an unbalanced rollback popping an empty stack), so abort like tagebp_update. */
static void tagebp_call_void(const char *method) {
	if (!tage_instance) return;
	PyObject *result = PyObject_CallMethod(tage_instance, method, NULL);
	if (!result) {
		PyErr_Print();
		fprintf(stderr, "[CE FATAL] tagebp: TAGE %s() raised\n", method);
		abort();
	}
	Py_DECREF(result);
}

void tagebp_checkpoint(void) { tagebp_call_void("checkpoint"); }
void tagebp_rollback(void)   { tagebp_call_void("rollback"); }

void tagebp_reset(void) {
	if (!tage_instance) return;
	PyObject *result = PyObject_CallMethod(tage_instance, "reset", NULL);
	if (!result) {
		fprintf(stderr, "[ERR] TAGE reset failed\n");
		PyErr_Print();
		__builtin_trap();
	}
	Py_DECREF(result);
}

void tagebp_destroy_instance() {
	if (tage_instance) {
		Py_DECREF(tage_instance);
		tage_instance = NULL;
	}
}

void python_finalize() {
    // Destroy any remaining instance
    tagebp_destroy_instance();

    if (python_initialized) {
        Py_FinalizeEx();
        python_initialized = 0;
    }
}

