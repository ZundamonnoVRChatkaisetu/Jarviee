import { keyframes } from '@emotion/react';

// Arc Reactor Glow Animation
export const arcReactorGlow = keyframes`
  0% {
    box-shadow: 0 0 5px rgba(100, 181, 246, 0.5), 0 0 10px rgba(100, 181, 246, 0.3);
  }
  50% {
    box-shadow: 0 0 15px rgba(100, 181, 246, 0.8), 0 0 20px rgba(100, 181, 246, 0.5);
  }
  100% {
    box-shadow: 0 0 5px rgba(100, 181, 246, 0.5), 0 0 10px rgba(100, 181, 246, 0.3);
  }
`;

// Holographic Flicker Animation
export const holographicFlicker = keyframes`
  0% {
    opacity: 1;
    text-shadow: 0 0 5px rgba(77, 208, 225, 0.8), 0 0 8px rgba(77, 208, 225, 0.5);
  }
  3% {
    opacity: 0.8;
    text-shadow: 0 0 4px rgba(77, 208, 225, 0.6), 0 0 6px rgba(77, 208, 225, 0.4);
  }
  5% {
    opacity: 1;
    text-shadow: 0 0 5px rgba(77, 208, 225, 0.8), 0 0 8px rgba(77, 208, 225, 0.5);
  }
  30% {
    opacity: 1;
    text-shadow: 0 0 5px rgba(77, 208, 225, 0.8), 0 0 8px rgba(77, 208, 225, 0.5);
  }
  33% {
    opacity: 0.9;
    text-shadow: 0 0 4px rgba(77, 208, 225, 0.7), 0 0 7px rgba(77, 208, 225, 0.4);
  }
  35% {
    opacity: 1;
    text-shadow: 0 0 5px rgba(77, 208, 225, 0.8), 0 0 8px rgba(77, 208, 225, 0.5);
  }
  100% {
    opacity: 1;
    text-shadow: 0 0 5px rgba(77, 208, 225, 0.8), 0 0 8px rgba(77, 208, 225, 0.5);
  }
`;

// System startup animation
export const systemStartup = keyframes`
  0% {
    opacity: 0;
    transform: scale(0.9);
  }
  20% {
    opacity: 0.3;
    transform: scale(0.92);
  }
  40% {
    opacity: 0.5;
    transform: scale(0.95);
  }
  60% {
    opacity: 0.7;
    transform: scale(0.97);
  }
  80% {
    opacity: 0.9;
    transform: scale(0.99);
  }
  100% {
    opacity: 1;
    transform: scale(1);
  }
`;

// Radial expand animation for startup
export const radialExpand = keyframes`
  0% {
    clip-path: circle(0% at center);
    opacity: 0;
  }
  100% {
    clip-path: circle(150% at center);
    opacity: 1;
  }
`;

// Typewriter effect animation
export const blink = keyframes`
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0;
  }
`;

// Pulse animation for notifications
export const pulse = keyframes`
  0% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.05);
  }
  100% {
    transform: scale(1);
  }
`;

// Progress bar animation
export const progress = keyframes`
  0% {
    width: 0%;
  }
  100% {
    width: 100%;
  }
`;

// Loading rings animation
export const loadingRings = keyframes`
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
`;

// Float animation for holographic elements
export const float = keyframes`
  0% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-10px);
  }
  100% {
    transform: translateY(0);
  }
`;

// Slide in from right
export const slideInRight = keyframes`
  0% {
    transform: translateX(100%);
    opacity: 0;
  }
  100% {
    transform: translateX(0);
    opacity: 1;
  }
`;

// Slide in from left
export const slideInLeft = keyframes`
  0% {
    transform: translateX(-100%);
    opacity: 0;
  }
  100% {
    transform: translateX(0);
    opacity: 1;
  }
`;

// Fade in animation
export const fadeIn = keyframes`
  0% {
    opacity: 0;
  }
  100% {
    opacity: 1;
  }
`;

// Fade out animation
export const fadeOut = keyframes`
  0% {
    opacity: 1;
  }
  100% {
    opacity: 0;
  }
`;

// Data scanning animation
export const dataScan = keyframes`
  0% {
    background-position: 0% 0%;
  }
  100% {
    background-position: 100% 0%;
  }
`;
