#pragma once

#include "ini.h"
#include <fcntl.h>
#include <math.h>
#include <unistd.h>

typedef struct {
    FILE* file;
    const char* path;
} TrainingMetrics;

void metrics_fail(TrainingMetrics* metrics, const char* operation) {
    fprintf(stderr, "training metrics: cannot %s %s: %s\n",
        operation, metrics->path, strerror(errno));
    exit(1);
}

TrainingMetrics metrics_open(const char* path) {
    TrainingMetrics metrics = {.file = NULL, .path = path};
    if (!path || !*path || !strcmp(path, "None")) {
        return metrics;
    }
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) {
        metrics_fail(&metrics, "create");
    }
    metrics.file = fdopen(fd, "w");
    return metrics;
}

void metrics_quoted(FILE* file, const char* value) {
    fputc('"', file);
    for (const unsigned char* p = (const unsigned char*)value; *p; ++p) {
        if (*p == '"' || *p == '\\') {
            fputc('\\', file);
            fputc(*p, file);
        } else if (*p < 0x20) {
            fprintf(file, "\\u%04x", *p);
        } else {
            fputc(*p, file);
        }
    }
    fputc('"', file);
}

void metrics_dict(FILE* file, const Dict* dict, int config) {
    fputc('{', file);
    for (int i = 0; i < dict->size; ++i) {
        if (i) {
            fputc(',', file);
        }
        const DictItem* item = &dict->items[i];
        metrics_quoted(file, item->key);
        fputc(':', file);
        if (config && item->str) {
            metrics_quoted(file, item->str);
        } else if (isfinite(item->value)) {
            fprintf(file, "%.17g", item->value);
        } else {
            fputs("null", file);
        }
    }
    fputc('}', file);
}

void metrics_flush(TrainingMetrics* metrics) {
    fputs("}\n", metrics->file);
    if (ferror(metrics->file) || fflush(metrics->file)) {
        metrics_fail(metrics, "write");
    }
}

void metrics_run(TrainingMetrics* metrics, const Ini* ini, int world_size) {
    if (!metrics->file) {
        return;
    }
    fprintf(metrics->file, "{\"schema\":\"puffer.training_metrics.v1\","
        "\"event\":\"run\",\"world_size\":%d,\"config\":{", world_size);
    for (int s = 0; s < ini->num_sections; ++s) {
        if (s) {
            fputc(',', metrics->file);
        }
        metrics_quoted(metrics->file, ini->sections[s].name);
        fputc(':', metrics->file);
        metrics_dict(metrics->file, &ini->sections[s], 1);
    }
    fputc('}', metrics->file);
    metrics_flush(metrics);
}

void metrics_log(TrainingMetrics* metrics, const Dict* values) {
    if (!metrics->file) {
        return;
    }
    fputs("{\"schema\":\"puffer.training_metrics.v1\",\"event\":\"metrics\",\"metrics\":",
        metrics->file);
    metrics_dict(metrics->file, values, 0);
    metrics_flush(metrics);
}

void metrics_close(TrainingMetrics* metrics) {
    if (metrics->file && fclose(metrics->file)) {
        metrics_fail(metrics, "close");
    }
    metrics->file = NULL;
}
