/*
  ==============================================================================

   This file is part of the JUCE tutorials.
   Copyright (c) 2020 - Raw Material Software Limited

   The code included in this file is provided under the terms of the ISC license
   http://www.isc.org/downloads/software-support-policy/isc-license. Permission
   To use, copy, modify, and/or distribute this software for any purpose with or
   without fee is hereby granted provided that the above copyright notice and
   this permission notice appear in all copies.

   THE SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY, AND ALL WARRANTIES,
   WHETHER EXPRESSED OR IMPLIED, INCLUDING MERCHANTABILITY AND FITNESS FOR
   PURPOSE, ARE DISCLAIMED.

  ==============================================================================
*/

/*******************************************************************************
 The block below describes the properties of this PIP. A PIP is a short snippet
 of code that can be read by the Projucer and used to generate a JUCE project.

 BEGIN_JUCE_PIP_METADATA

 name:             SDRNN_7to7
 version:          1.0.0
 vendor:           JUCE
 website:          http://juce.com
 description:      Segment Display Recognition Neural Network (7 to 7 model).

 dependencies:     juce_core, juce_data_structures, juce_events, juce_graphics,
                   juce_gui_basics
 exporters:        xcode_mac, vs2019, linux_make

 type:             Component
 mainClass:        MainContentComponent

 useLocalCopy:     1

 END_JUCE_PIP_METADATA

*******************************************************************************/


#pragma once
#include "MLP.h"
#include <string>

#define WIN_W 750
#define WIN_H 350

#define O1X  20
#define O1Y  20
#define O2X 470 
#define VSW 72
#define VSL 100
#define HSW 50
#define HSL 100
#define V1X 0+O1X
#define V2X V1X+VSW+VSL
#define H1Y 0+O1Y
#define H2Y H1Y+HSW+VSL-15
#define H3Y H2Y+HSW+VSL-15
#define SLaX V1X+VSW 
#define SLaY H1Y
#define SLbX V2X
#define SLbY H1Y+HSW
#define SLcX V2X
#define SLcY H2Y+HSW
#define SLdX V1X+VSW 
#define SLdY H3Y
#define SLeX V1X
#define SLeY H2Y+HSW
#define SLfX V1X
#define SLfY H1Y+HSW
#define SLgX V1X+VSW 
#define SLgY H2Y
#define SEGW 25
#define SEGL 100
#define aX SLaX
#define aY SLaY+((HSW*5)/12)
#define bX SLbX
#define bY SLbY
#define cX SLcX
#define cY SLcY
#define dX SLdX
#define dY SLdY+((HSW*5)/12)
#define eX SLeX+((VSW*5)/8)
#define eY SLeY
#define fX SLfX+((VSW*5)/8)
#define fY SLfY
#define gX SLgX
#define gY SLgY+((HSW*5)/12)

#define OFFSET 230

//==============================================================================
class MainContentComponent : public juce::Component,
                             public juce::Slider::Listener,
                             public juce::Button::Listener,
                             public juce::TextEditor::Listener{
public:
    //==============================================================================
    MainContentComponent(){
        srand(time(NULL));
        rand();
        sdrnn = new MultiLayerPerceptron({7,7,7});
        LookAndFeel *l = &getLookAndFeel();
        l->setColour(juce::Slider::thumbColourId,juce::Colour(110,110,110));
        l->setColour(juce::Slider::textBoxOutlineColourId,Colour(240,240,240));
        l->setColour(juce::Slider::textBoxTextColourId,juce::Colours::black);
        l->setColour(juce::Slider::backgroundColourId,juce::Colour(OFFSET,OFFSET,OFFSET));
        l->setColour(juce::Slider::trackColourId, juce::Colour(OFFSET, OFFSET, OFFSET));
        l->setColour(juce::TextButton::buttonColourId, juce::Colour(OFFSET, OFFSET, OFFSET));
        l->setColour(juce::TextButton::textColourOffId, juce::Colours::black);
        l->setColour(juce::TextButton::textColourOnId, juce::Colours::black);
        l->setColour(juce::Label::textColourId, juce::Colours::black);
        l->setColour(juce::Label::backgroundColourId, juce::Colour(240,240,240));
        l->setColour(juce::TextEditor::ColourIds::textColourId, juce::Colours::black);
        l->setColour(juce::TextEditor::ColourIds::backgroundColourId, juce::Colours::white);

        addAndMakeVisible(lbl_epochs_txt);
        addAndMakeVisible(entry_epochs);
        lbl_epochs_txt.setText("Epochs to Train:", no);
        entry_epochs.setText("10");
        
        addAndMakeVisible(btn_train);
        btn_train.setButtonText("Train some more");
        btn_train.addListener(this);

        addAndMakeVisible(lbl_err_txt);
        addAndMakeVisible(lbl_err);
        lbl_err_txt.setText("Training Error:",no);
        lbl_err.setText("---",no);
        lbl_err.setJustificationType(juce::Justification::centred);

        addAndMakeVisible(lbl_tepochs_txt);
        addAndMakeVisible(lbl_tepochs);
        lbl_tepochs_txt.setText("Epochs so far:",no);
        lbl_tepochs.setText("0",no);
        lbl_tepochs.setJustificationType(juce::Justification::centred);

        addAndMakeVisible(btn_reset);
        btn_reset.setButtonText("Reset");
        btn_reset.addListener(this);
        
        addAndMakeVisible (slider_a);
        slider_a.setRange (0.0, 1.0);
        slider_a.addListener (this);
        slider_a.setTextBoxStyle(juce::Slider::TextBoxAbove,true,60,20);
        slider_a.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_b);
        slider_b.setRange(0.0,1.0);
        slider_b.addListener(this);
        slider_b.setSliderStyle(juce::Slider::LinearVertical);
        slider_b.setTextBoxStyle(juce::Slider::TextBoxRight,true,60,20);
        slider_b.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_c);
        slider_c.setRange(0.0,1.0);
        slider_c.addListener(this);
        slider_c.setSliderStyle(juce::Slider::LinearVertical);
        slider_c.setTextBoxStyle(juce::Slider::TextBoxRight,true,60,20);
        slider_c.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_d);
        slider_d.setRange(0.0,1.0);
        slider_d.addListener(this);
        slider_d.setTextBoxStyle(juce::Slider::TextBoxAbove,true,60,20);
        slider_d.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_e);
        slider_e.setRange(0.0,1.0);
        slider_e.addListener(this);
        slider_e.setSliderStyle(juce::Slider::LinearVertical);
        slider_e.setTextBoxStyle(juce::Slider::TextBoxLeft,true,60,20);
        slider_e.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_f);
        slider_f.setRange(0.0,1.0);
        slider_f.addListener(this);
        slider_f.setSliderStyle(juce::Slider::LinearVertical);
        slider_f.setTextBoxStyle(juce::Slider::TextBoxLeft,true,60,20);
        slider_f.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_g);
        slider_g.setRange(0.0,1.0);
        slider_g.addListener(this);
        slider_g.setTextBoxStyle(juce::Slider::TextBoxAbove,true,60,20);
        slider_g.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible (slider_a2);
        slider_a2.setRange (0.0, 1.0);
        slider_a2.addListener (this);
        slider_a2.setTextBoxStyle(juce::Slider::TextBoxAbove,true,60,20);
        slider_a2.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_b2);
        slider_b2.setRange(0.0,1.0);
        slider_b2.addListener(this);
        slider_b2.setSliderStyle(juce::Slider::LinearVertical);
        slider_b2.setTextBoxStyle(juce::Slider::TextBoxRight,true,60,20);
        slider_b2.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_c2);
        slider_c2.setRange(0.0,1.0);
        slider_c2.addListener(this);
        slider_c2.setSliderStyle(juce::Slider::LinearVertical);
        slider_c2.setTextBoxStyle(juce::Slider::TextBoxRight,true,60,20);
        slider_c2.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_d2);
        slider_d2.setRange(0.0,1.0);
        slider_d2.addListener(this);
        slider_d2.setTextBoxStyle(juce::Slider::TextBoxAbove,true,60,20);
        slider_d2.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_e2);
        slider_e2.setRange(0.0,1.0);
        slider_e2.addListener(this);
        slider_e2.setSliderStyle(juce::Slider::LinearVertical);
        slider_e2.setTextBoxStyle(juce::Slider::TextBoxLeft,true,60,20);
        slider_e2.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_f2);
        slider_f2.setRange(0.0,1.0);
        slider_f2.addListener(this);
        slider_f2.setSliderStyle(juce::Slider::LinearVertical);
        slider_f2.setTextBoxStyle(juce::Slider::TextBoxLeft,true,60,20);
        slider_f2.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_g2);
        slider_g2.setRange(0.0,1.0);
        slider_g2.addListener(this);
        slider_g2.setTextBoxStyle(juce::Slider::TextBoxAbove,true,60,20);
        slider_g2.setNumDecimalPlacesToDisplay(2);

        setSize(WIN_W, WIN_H);
        run_ann();
    }


#define DX   90
#define DY   30
#define LEN  90
#define HEI  25 
#define X0  290
#define X1 (X0+DX)
#define Y0   30
#define Y1 (Y0+DY)
#define Y2 (Y1+DY)
#define Y3 (Y2+DY)
#define YR (Y3+DY)

    void resized() override{
        slider_a.setBounds(SLaX,SLaY,HSL,HSW);
        slider_b.setBounds(SLbX,SLbY,VSW,VSL);
        slider_c.setBounds(SLcX,SLcY,VSW,VSL);
        slider_d.setBounds(SLdX,SLdY,HSL,HSW);
        slider_e.setBounds(SLeX,SLeY,VSW,VSL);
        slider_f.setBounds(SLfX,SLfY,VSW,VSL);
        slider_g.setBounds(SLgX,SLgY,HSL,HSW);

        lbl_epochs_txt.setBounds  (X0, Y0, LEN, HEI);
        entry_epochs.setBounds    (X1, Y0, LEN, HEI);
        btn_train.setBounds       (X1, Y1, LEN, HEI);
        lbl_err_txt.setBounds     (X0, Y2, LEN, HEI);
        lbl_err.setBounds         (X1, Y2, LEN, HEI);
        lbl_tepochs_txt.setBounds (X0, Y3, LEN, HEI);
        lbl_tepochs.setBounds     (X1, Y3, LEN, HEI);
        btn_reset.setBounds       (X0, YR, LEN, HEI);
       

        slider_a2.setBounds(O2X + SLaX, SLaY, HSL, HSW);
        slider_b2.setBounds(O2X + SLbX, SLbY, VSW, VSL);
        slider_c2.setBounds(O2X + SLcX, SLcY, VSW, VSL);
        slider_d2.setBounds(O2X + SLdX, SLdY, HSL, HSW);
        slider_e2.setBounds(O2X + SLeX, SLeY, VSW, VSL);
        slider_f2.setBounds(O2X + SLfX, SLfY, VSW, VSL);
        slider_g2.setBounds(O2X + SLgX, SLgY, HSL, HSW);

        setSize(WIN_W,WIN_H);
    }

    void sliderValueChanged (juce::Slider* slider) override {
        Colour c(    (int)(slider->getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider->getValue()* OFFSET),
            OFFSET - (int)(slider->getValue()* OFFSET)    );

        slider->setColour(juce::Slider::backgroundColourId,c);
        slider->setColour(juce::Slider::trackColourId,c);
        run_ann();
        repaint();
    }

    void run_ann(){
        std::vector <double> x;
        x.push_back(slider_a.getValue());
        x.push_back(slider_b.getValue());
        x.push_back(slider_c.getValue());
        x.push_back(slider_d.getValue());
        x.push_back(slider_e.getValue());
        x.push_back(slider_f.getValue());
        x.push_back(slider_g.getValue());
        std::vector <double> out = sdrnn->run(x);
        slider_a2.setValue(out[0]);
        slider_b2.setValue(out[1]);
        slider_c2.setValue(out[2]);
        slider_d2.setValue(out[3]);
        slider_e2.setValue(out[4]);
        slider_f2.setValue(out[5]);
        slider_g2.setValue(out[6]);
    }

    void train_ann(){
        double MSE;
        int epochs = entry_epochs.getText().getIntValue();
        for (int i = 0; i < epochs; i++){
            MSE = 0.0;
            MSE += sdrnn->bp({1,1,1,1,1,1,0}, {1,1,1,1,1,1,0}); //0 pattern
            MSE += sdrnn->bp({0,1,1,0,0,0,0}, {0,1,1,0,0,0,0}); //1 pattern
            MSE += sdrnn->bp({1,1,0,1,1,0,1}, {1,1,0,1,1,0,1}); //2 pattern
            MSE += sdrnn->bp({1,1,1,1,0,0,1}, {1,1,1,1,0,0,1}); //3 pattern
            MSE += sdrnn->bp({0,1,1,0,0,1,1}, {0,1,1,0,0,1,1}); //4 pattern
            MSE += sdrnn->bp({1,0,1,1,0,1,1}, {1,0,1,1,0,1,1}); //5 pattern
            MSE += sdrnn->bp({1,0,1,1,1,1,1}, {1,0,1,1,1,1,1}); //6 pattern
            MSE += sdrnn->bp({1,1,1,0,0,0,0}, {1,1,1,0,0,0,0}); //7 pattern
            MSE += sdrnn->bp({1,1,1,1,1,1,1}, {1,1,1,1,1,1,1}); //8 pattern
            MSE += sdrnn->bp({1,1,1,1,0,1,1}, {1,1,1,1,0,1,1}); //9 pattern
        }
        MSE /= 10.0;
        lbl_err.setText(std::to_string(MSE), no);
        tepochs += epochs;
        lbl_tepochs.setText(std::to_string(tepochs), no);
        run_ann();
    }

    void buttonClicked(juce::Button* button) override{
        if (button == &btn_train)
            train_ann();
        if (button == &btn_reset){
            delete(sdrnn);
            sdrnn = new MultiLayerPerceptron({7,7,7});
            tepochs = 0;
            lbl_err.setText("---", no);
            lbl_tepochs.setText(std::to_string(tepochs), no);
            run_ann();
        }
    }

    void paint(juce::Graphics& g) override {
        g.fillAll(juce::Colour(240,240,240));
        
        g.setColour(juce::Colour(
            (int)(slider_a.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_a.getValue()* OFFSET),
            OFFSET - (int)(slider_a.getValue()* OFFSET)     ));
        g.fillRect(aX,aY,SEGL,SEGW);

        g.setColour(juce::Colour(
            (int)(slider_b.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_b.getValue()* OFFSET),
            OFFSET - (int)(slider_b.getValue()* OFFSET)     ));
        g.fillRect(bX,bY,SEGW,SEGL);

        g.setColour(juce::Colour(
            (int)(slider_c.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_c.getValue()* OFFSET),
            OFFSET - (int)(slider_c.getValue()* OFFSET)     ));
        g.fillRect(cX,cY,SEGW,SEGL);

        g.setColour(juce::Colour(
            (int)(slider_d.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_d.getValue()* OFFSET),
            OFFSET - (int)(slider_d.getValue()* OFFSET)     ));
        g.fillRect(dX,dY,SEGL,SEGW);

        g.setColour(juce::Colour(
            (int)(slider_e.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_e.getValue()* OFFSET),
            OFFSET - (int)(slider_e.getValue()* OFFSET)     ));
        g.fillRect(eX,eY,SEGW,SEGL);

        g.setColour(juce::Colour(
            (int)(slider_f.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_f.getValue()* OFFSET),
            OFFSET - (int)(slider_f.getValue()* OFFSET)     ));
        g.fillRect(fX,fY,SEGW,SEGL);

        g.setColour(juce::Colour(
            (int)(slider_g.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_g.getValue()* OFFSET),
            OFFSET - (int)(slider_g.getValue()* OFFSET)     ));
        g.fillRect(gX,gY,SEGL,SEGW);

        
        g.setColour(juce::Colour(
            (int)(slider_a2.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_a2.getValue()* OFFSET),
            OFFSET - (int)(slider_a2.getValue()* OFFSET)     ));
        g.fillRect(O2X+aX,aY,SEGL,SEGW);

        g.setColour(juce::Colour(
            (int)(slider_b2.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_b2.getValue()* OFFSET),
            OFFSET - (int)(slider_b2.getValue()* OFFSET)     ));
        g.fillRect(O2X+bX,bY,SEGW,SEGL);

        g.setColour(juce::Colour(
            (int)(slider_c2.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_c2.getValue()* OFFSET),
            OFFSET - (int)(slider_c2.getValue()* OFFSET)     ));
        g.fillRect(O2X+cX,cY,SEGW,SEGL);

        g.setColour(juce::Colour(
            (int)(slider_d2.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_d2.getValue()* OFFSET),
            OFFSET - (int)(slider_d2.getValue()* OFFSET)     ));
        g.fillRect(O2X+dX,dY,SEGL,SEGW);

        g.setColour(juce::Colour(
            (int)(slider_e2.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_e2.getValue()* OFFSET),
            OFFSET - (int)(slider_e2.getValue()* OFFSET)     ));
        g.fillRect(O2X+eX,eY,SEGW,SEGL);

        g.setColour(juce::Colour(
            (int)(slider_f2.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_f2.getValue()* OFFSET),
            OFFSET - (int)(slider_f2.getValue()* OFFSET)     ));
        g.fillRect(O2X+fX,fY,SEGW,SEGL);

        g.setColour(juce::Colour(
            (int)(slider_g2.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_g2.getValue()* OFFSET),
            OFFSET - (int)(slider_g2.getValue()* OFFSET)     ));
        g.fillRect(O2X+gX,gY,SEGL,SEGW);

    }

    MultiLayerPerceptron * sdrnn;
    juce::NotificationType no = juce::dontSendNotification;
    int tepochs = 0;

private:
    juce::Slider slider_a;
    juce::Slider slider_b;
    juce::Slider slider_c;
    juce::Slider slider_d;
    juce::Slider slider_e;
    juce::Slider slider_f;
    juce::Slider slider_g;

    juce::Label lbl_epochs_txt;
    juce::TextEditor entry_epochs;
    juce::TextButton btn_train;
    juce::Label lbl_err_txt; 
    juce::Label lbl_err;
    juce::Label lbl_tepochs_txt;
    juce::Label lbl_tepochs;
    juce::TextButton btn_reset;

    juce::Slider slider_a2;
    juce::Slider slider_b2;
    juce::Slider slider_c2;
    juce::Slider slider_d2;
    juce::Slider slider_e2;
    juce::Slider slider_f2;
    juce::Slider slider_g2;
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainContentComponent)
};
