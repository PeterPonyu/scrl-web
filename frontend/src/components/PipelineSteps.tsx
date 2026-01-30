'use client';

import React from 'react';
import { Check, Circle, Loader2 } from 'lucide-react';

interface Step {
  id: string;
  name: string;
  description: string;
}

interface PipelineStepsProps {
  steps: Step[];
  currentStep: string;
  completedSteps: string[];
  isProcessing: boolean;
}

export function PipelineSteps({ steps, currentStep, completedSteps, isProcessing }: PipelineStepsProps) {
  return (
    <nav aria-label="Progress">
      <ol className="space-y-4">
        {steps.map((step, index) => {
          const isCompleted = completedSteps.includes(step.id);
          const isCurrent = currentStep === step.id;
          
          return (
            <li key={step.id} className="relative">
              {index !== steps.length - 1 && (
                <div
                  className={`absolute left-4 top-8 w-0.5 h-full -translate-x-1/2 ${
                    isCompleted ? 'bg-primary-500' : 'bg-gray-200'
                  }`}
                />
              )}
              <div className="flex items-start gap-3">
                <div
                  className={`
                    flex items-center justify-center w-8 h-8 rounded-full flex-shrink-0
                    ${isCompleted ? 'bg-primary-500' : isCurrent ? 'bg-primary-100 border-2 border-primary-500' : 'bg-gray-100'}
                  `}
                >
                  {isCompleted ? (
                    <Check className="w-4 h-4 text-white" />
                  ) : isCurrent && isProcessing ? (
                    <Loader2 className="w-4 h-4 text-primary-600 animate-spin" />
                  ) : (
                    <Circle className={`w-4 h-4 ${isCurrent ? 'text-primary-600' : 'text-gray-400'}`} />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <p className={`text-sm font-medium ${isCurrent ? 'text-primary-600' : isCompleted ? 'text-gray-900' : 'text-gray-500'}`}>
                    {step.name}
                  </p>
                  <p className="text-xs text-gray-400 mt-0.5">
                    {step.description}
                  </p>
                </div>
              </div>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
